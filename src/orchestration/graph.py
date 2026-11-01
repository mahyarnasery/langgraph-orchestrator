from __future__ import annotations

import ast
import csv
from graphlib import CycleError, TopologicalSorter
import io
import json
import re
import tomllib
from pathlib import Path

from langgraph.graph import END, StateGraph

from src.agents.foreman import ForemanAgent
from src.agents.planner import PlannerAgent
from src.harnesses.aider import AiderHarness, list_workspace_files, list_workspace_py_files

from .checkpoint import (
    clear_checkpoint,
    load_checkpoint,
    reconcile_checkpoint_with_disk,
    save_checkpoint,
    update_checkpoint_stage_metadata,
)
from .ledger import (
    ensure_class_wrapper,
    extract_all_ast_symbol_names,
    extract_python_signatures,
    format_ledger_for_prompt,
    merge_staged_snippets_ast,
    record_ledger_entry,
    upsert_staged_snippet,
)
from .schemas import (
    CycleResult,
    ForemanReview,
    Plan,
    PlannerReview,
    StageGoal,
    Task,
    TaskBatch,
    TaskResult,
    ValidationResult,
    WorkerProfile,
)
from .state import WorkflowState
from .validator import emit_pytest_suite, validate_batch
from .workspace import (
    DEFAULT_WORKSPACE_ROOT,
    REPO_ROOT,
    get_access_mode,
    get_external_read_files,
    get_staging_dir,
    get_workspace_root,
    normalize_workspace_rel_path,
    set_active_workspace,
)

WORKER_WORKSPACE = DEFAULT_WORKSPACE_ROOT
MAX_FOREMAN_REPAIRS = 2
MAX_WORKER_REBUILDS = 3


def configure_loop_limits(
    *,
    max_repairs: int | None = None,
    max_rebuilds: int | None = None,
) -> None:
    """Override Foreman self-repair and Worker rebuild loop limits at runtime."""
    global MAX_FOREMAN_REPAIRS, MAX_WORKER_REBUILDS
    if max_repairs is not None and max_repairs >= 0:
        MAX_FOREMAN_REPAIRS = max_repairs
    if max_rebuilds is not None and max_rebuilds >= 0:
        MAX_WORKER_REBUILDS = max_rebuilds


def _normalize_workspace_path(path: str) -> str:
    """Normalize a logical workspace path into a clean relative POSIX path."""
    return normalize_workspace_rel_path(path)


def _sandbox_repo_path(path: str) -> str:
    """Convert a logical TaskBatch path into a workspace-relative path for tool calls."""
    normalized = _normalize_workspace_path(path)
    if normalized in ("", ".", ".."):
        raise ValueError(f"Invalid Worker path: {path}")
    return normalized


def _summarize_non_python_asset(path: Path, source: str) -> str:
    """Build a compact summary including actual values for small JSON/CSV/TOML assets so Foreman writes accurate assertions."""
    suffix = path.suffix.lower()
    if suffix == ".json":
        try:
            data = json.loads(source)
            compact_json = json.dumps(data)
            if len(compact_json) <= 240:
                return f"JSON content: {compact_json}"
            if isinstance(data, dict):
                return f"JSON object (keys: {list(data.keys())[:8]})"
            if isinstance(data, list):
                return f"JSON array ({len(data)} items)"
        except Exception:
            return "JSON (unparsed)"
    elif suffix == ".toml":
        try:
            data = tomllib.loads(source)
            return f"TOML (keys: {list(data.keys())[:8]})"
        except Exception:
            return "TOML (unparsed)"
    elif suffix == ".csv":
        try:
            rows = [
                r for r in csv.reader(io.StringIO(source.strip()))
                if any(c.strip() for c in r)
            ]
            if rows:
                preview = "; ".join(",".join(r) for r in rows[:6])
                return f"CSV ({len(rows)} rows: {preview})"
        except Exception:
            return "CSV (unparsed)"
    lines = [ln for ln in source.splitlines() if ln.strip()]
    return f"{suffix.lstrip('.').upper() or 'ASSET'} ({len(lines)} non-empty lines)"


def _build_repository_map() -> str:
    """Build a compact map of existing Python modules and non-Python config/data assets in the active workspace."""
    ws_root = get_workspace_root()
    ws_files = list_workspace_files()
    ext_files = (
        [p for p in get_external_read_files() if p.is_file()]
        if get_access_mode() in ("read_external", "allow_external")
        else []
    )
    if not ws_files and not ext_files:
        return f"[No files currently in workspace {ws_root.name}/]"

    entries: list[str] = []
    for fpath in ws_files:
        rel_path = str(fpath.relative_to(ws_root))
        try:
            source = fpath.read_text(encoding="utf-8")
            if not source.strip():
                if fpath.name == "__init__.py":
                    entries.append(f"- {rel_path} (package marker)")
                else:
                    entries.append(f"- {rel_path} (empty file)")
                continue

            if fpath.suffix.lower() == ".py":
                sigs = extract_python_signatures(source)
                summary = "; ".join(sigs) if sigs else f"{len(source.splitlines())} lines"
            else:
                summary = _summarize_non_python_asset(fpath, source)
            entries.append(f"- {rel_path}: {summary}")
        except Exception:
            entries.append(f"- {rel_path} (unparsed)")

    if get_access_mode() in ("read_external", "allow_external"):
        for ext_file in get_external_read_files():
            if ext_file.is_file():
                try:
                    ext_src = ext_file.read_text(encoding="utf-8")
                    if ext_file.suffix.lower() == ".py":
                        sigs = extract_python_signatures(ext_src)
                        summary = "; ".join(sigs) if sigs else f"{len(ext_src.splitlines())} lines"
                    else:
                        summary = _summarize_non_python_asset(ext_file, ext_src)
                    entries.append(f"- [EXTERNAL READ-ONLY] {ext_file}: {summary}")
                except Exception:
                    pass
    return f"Existing Workspace Repository Map ({ws_root}, access_mode={get_access_mode()}):\n" + "\n".join(entries)


def _enrich_architecture_context(architecture_context: str) -> str:
    repo_map = _build_repository_map()
    return f"{architecture_context}\n\n{repo_map}"


_PYTHON_BUILTINS = set(dir(__builtins__)) if isinstance(__builtins__, dict) else set(dir(__builtins__))


def _invalid_symbol_names_for_task(task: Task) -> set[str]:
    """Return Python builtins plus all directory/file stems from `task.relevant_files`."""
    invalid = set(_PYTHON_BUILTINS) | {
        "class", "function", "method", "file", "script", "module",
        "the", "a", "an", "services", "domain", "models", "core",
        "config", "data", "tests", "main", "app", "utils", "helpers",
    }
    for rf in task.relevant_files:
        p = Path(_normalize_workspace_path(rf))
        invalid.add(p.stem)
        invalid.add(p.name)
        for part in p.parts:
            invalid.add(part)
            invalid.add(Path(part).stem)
    return invalid


def _infer_task_symbol(task: Task, multi_task_file: bool) -> str | None:
    """
    Validate or infer `target_symbol` on a Python task:
    - If `multi_task_file` is False (only 1 task targets this file), clear any bogus
      module/directory `target_symbol` (like `'services'`) and keep `target_symbol`
      only if it is a genuine function or `ClassName.method` name.
    - Never match path prefixes before `/` (e.g. `services/ingestion.py`).
    """
    if task.relevant_files and not task.relevant_files[0].endswith(".py"):
        return None

    invalid_names = _invalid_symbol_names_for_task(task)

    if task.target_symbol and task.target_symbol.strip():
        cand = task.target_symbol.split("(")[0].strip().strip(".")
        if "/" not in cand and cand not in invalid_names and cand.split(".")[0] not in _PYTHON_BUILTINS:
            return cand
        task.target_symbol = None

    # Do NOT guess a single target_symbol when an entire multi-function module is in 1 task
    if not multi_task_file:
        return None

    combined_text = f"{task.objective}\n" + "\n".join(task.acceptance_criteria)
    cls_method_match = re.search(
        r"\b([A-Z][a-zA-Z0-9_]*\.[a-zA-Z_][a-zA-Z0-9_]*)\b",
        combined_text,
    )
    if cls_method_match:
        return cls_method_match.group(1)

    for crit in task.acceptance_criteria:
        m = re.search(r"\b(?:def|class)\s+([a-zA-Z_][a-zA-Z0-9_]*)\b", crit)
        if m and m.group(1) not in invalid_names:
            return m.group(1)

    for token in re.findall(r"`([a-zA-Z_][a-zA-Z0-9_.]*)(?:\([^`]*\))?`", task.objective):
        if (
            "/" not in token
            and not token.endswith((".py", ".json", ".csv", ".toml", ".yaml", ".yml", ".ini", ".sql", ".md", ".txt"))
            and token not in invalid_names
        ):
            return token

    m_obj = re.search(
        r"\b(?:implement|define|create)\s+(?:the\s+)?(?:class\s+|function\s+|method\s+)?([a-zA-Z_][a-zA-Z0-9_]*)(?![a-zA-Z0-9_./])",
        task.objective,
        re.IGNORECASE,
    )
    if m_obj:
        cand_obj = m_obj.group(1).strip(".")
        if cand_obj not in invalid_names:
            return cand_obj

    for assertion in task.test_assertions:
        for call_name in re.findall(r"\b([a-zA-Z_][a-zA-Z0-9_]*)\s*\(", assertion):
            if call_name not in invalid_names and call_name != "assert":
                return call_name

    return None


def _is_redundant_empty_file_task(task: Task) -> bool:
    """Return True if a task only exists to create an empty file or empty __init__.py."""
    scope_lower = task.scope.strip().lower()
    obj_lower = task.objective.strip().lower()

    if (
        scope_lower in ("file creation", "create file", "empty file creation")
        and len(task.acceptance_criteria) == 1
        and "exists" in task.acceptance_criteria[0].lower()
    ):
        return True

    if len(task.relevant_files) == 1 and Path(task.relevant_files[0]).name == "__init__.py":
        joined_criteria = " ".join(task.acceptance_criteria).lower()
        has_code_req = any(
            kw in joined_criteria
            for kw in ("def ", "class ", "import ", "__all__", "export")
        )
        if not has_code_req and not task.test_assertions and (
            "empty" in obj_lower
            or "empty" in joined_criteria
            or "exists" in joined_criteria
            or "initialize" in obj_lower
            or "package" in obj_lower
        ):
            return True

    return False


def _sort_execution_order_topologically(
    tasks: list[Task],
    initial_order: list[str],
) -> list[str]:
    """
    Deterministically sort task IDs in dependency order using TopologicalSorter:
    - Schedules non-Python config/data tasks (`.json`, `.csv`, `.toml`, etc.) before Python tasks when at the same depth.
    - Ensures `ClassName.__init__` tasks run before other method tasks on the same file/class.
    - Ensures entry-point scripts (`main.py`, `app.py`) run after library/model tasks.
    """
    valid_ids = {t.task_id for t in tasks}
    rank = {tid: idx for idx, tid in enumerate(initial_order) if tid in valid_ids}
    for t in tasks:
        rank.setdefault(t.task_id, len(rank))

    asset_task_ids = {
        t.task_id
        for t in tasks
        if t.relevant_files and not t.relevant_files[0].endswith(".py")
    }

    init_task_by_file_class: dict[tuple[str, str], str] = {}
    init_task_by_file: dict[str, str] = {}
    for t in tasks:
        if not t.relevant_files or not t.target_symbol:
            continue
        primary = t.relevant_files[0]
        if t.target_symbol.endswith(".__init__"):
            cls_name = t.target_symbol.split(".", 1)[0]
            init_task_by_file_class[(primary, cls_name)] = t.task_id
            init_task_by_file[primary] = t.task_id
        elif t.target_symbol == "__init__":
            init_task_by_file[primary] = t.task_id

    non_entry_ids = {
        t.task_id
        for t in tasks
        if not (
            t.target_symbol is None
            and len(t.relevant_files) == 1
            and Path(t.relevant_files[0]).name in ("main.py", "app.py")
        )
    }
    referenced_as_dep = {
        dep for t in tasks for dep in t.dependencies if dep in valid_ids
    }

    graph_deps: dict[str, list[str]] = {}
    for t in sorted(tasks, key=lambda x: rank[x.task_id]):
        deps = [d for d in t.dependencies if d in valid_ids and d != t.task_id]

        # Ensure non-Python config/data files are created before Python modules
        if t.task_id not in asset_task_ids and asset_task_ids:
            for asset_tid in sorted(asset_task_ids, key=lambda tid: rank[tid]):
                if asset_tid not in deps:
                    deps.append(asset_tid)

        if t.relevant_files and t.target_symbol and not t.target_symbol.endswith("__init__"):
            primary = t.relevant_files[0]
            init_tid: str | None = None
            if "." in t.target_symbol:
                cls_name = t.target_symbol.split(".", 1)[0]
                init_tid = init_task_by_file_class.get((primary, cls_name))
            if init_tid is None:
                init_tid = init_task_by_file.get(primary)
            if init_tid and init_tid != t.task_id and init_tid not in deps:
                deps.append(init_tid)

        if (
            t.task_id not in non_entry_ids
            and t.task_id not in referenced_as_dep
            and non_entry_ids
        ):
            for lib_tid in sorted(non_entry_ids, key=lambda tid: rank[tid]):
                if lib_tid not in deps:
                    deps.append(lib_tid)

        graph_deps[t.task_id] = deps

    try:
        sorter = TopologicalSorter()
        for tid in sorted(valid_ids, key=lambda x: rank[x]):
            sorter.add(tid, *graph_deps.get(tid, []))
        sorter.prepare()
        ordered: list[str] = []
        while sorter.is_active():
            ready = sorted(sorter.get_ready(), key=lambda x: rank[x])
            ordered.extend(ready)
            sorter.done(*ready)
        return ordered
    except CycleError as exc:
        print(f"[WARN] Circular task dependency detected ({exc}); retaining initial order.")
        return sorted(valid_ids, key=lambda x: rank[x])



def _align_batch_with_plan_blueprints(batch: TaskBatch, plan: Plan | None) -> None:
    """
    Deterministically align Foreman's TaskBatch with the Planner's `file_blueprints`:
    - If a multi-symbol Python file was lumped into a single task by Foreman even though
      the Planner designed 2+ `SymbolSpec`s for that file, expand it into atomic per-symbol tasks.
    - If a task's `target_symbol` is missing or imprecise, match it against the Planner's
      `SymbolSpec`s for that file.
    """
    if plan is None or not getattr(plan, "file_blueprints", None):
        return

    blueprints_by_path = {
        _normalize_workspace_path(fb.path): fb
        for fb in plan.file_blueprints
        if _normalize_workspace_path(fb.path)
    }
    if not blueprints_by_path:
        return

    # Filter out any tasks that target files outside the Planner's file_blueprints
    # (prevents Foreman from prematurely adding Stage 2 files during Stage 1).
    allowed_plan_files = set(blueprints_by_path.keys())
    in_scope_tasks = [
        t for t in batch.tasks
        if t.relevant_files
        and all(_normalize_workspace_path(rf) in allowed_plan_files for rf in t.relevant_files)
    ]
    if in_scope_tasks:
        batch.tasks = in_scope_tasks
        batch.relevant_files = sorted(
            {
                _normalize_workspace_path(rf)
                for t in batch.tasks
                for rf in t.relevant_files
            }
        )

    tasks_by_file: dict[str, list[Task]] = {}
    for t in batch.tasks:
        if t.relevant_files:
            tasks_by_file.setdefault(_normalize_workspace_path(t.relevant_files[0]), []).append(t)

    new_tasks: list[Task] = []
    expanded_any = False

    for t in batch.tasks:
        primary = _normalize_workspace_path(t.relevant_files[0]) if t.relevant_files else ""
        fb = blueprints_by_path.get(primary)
        if fb is None or not primary.endswith(".py") or not fb.symbols:
            new_tasks.append(t)
            continue

        file_tasks = tasks_by_file.get(primary, [])
        # Case A: Planner designed 2+ symbols for this file, but Foreman emitted only 1 lumped task
        if len(fb.symbols) >= 2 and len(file_tasks) == 1:
            expanded_any = True
            prev_sub_id: str | None = None
            for idx, sym_spec in enumerate(fb.symbols, start=1):
                sub_id = f"{t.task_id}_{idx}"
                sub_deps = list(t.dependencies)
                if prev_sub_id and sym_spec.target_symbol.split(".")[-1] != "__init__":
                    sub_deps.append(prev_sub_id)
                # Keep relevant test_assertions that reference this symbol or class
                sym_short = sym_spec.target_symbol.split(".")[-1]
                cls_prefix = sym_spec.target_symbol.split(".")[0] if "." in sym_spec.target_symbol else ""
                sub_asserts = [
                    a for a in t.test_assertions
                    if sym_short in a or (sym_short == "__init__" and cls_prefix and cls_prefix in a)
                ]
                if not sub_asserts and idx == len(fb.symbols):
                    sub_asserts = list(t.test_assertions)
                new_tasks.append(
                    Task(
                        task_id=sub_id,
                        objective=f"Implement `{sym_spec.target_symbol}` in `{primary}`: {sym_spec.contract}",
                        target_symbol=sym_spec.target_symbol,
                        scope=f"Atomic symbol task from Planner blueprint ({primary})",
                        relevant_files=[primary],
                        dependencies=sub_deps,
                        constraints=list(fb.required_imports),
                        acceptance_criteria=[sym_spec.signature, sym_spec.contract],
                        test_assertions=sub_asserts,
                    )
                )
                if sym_spec.target_symbol.endswith(".__init__") or prev_sub_id is None:
                    prev_sub_id = sub_id
            continue

        # Case B: Match/correct `task.target_symbol` against the Planner's SymbolSpec list
        if len(file_tasks) == 1 and len(fb.symbols) == 1 and "." in fb.symbols[0].target_symbol:
            t.target_symbol = fb.symbols[0].target_symbol.split(".")[0]
        else:
            valid_planned_symbols = [s.target_symbol for s in fb.symbols]
            if t.target_symbol not in valid_planned_symbols:
                # Check task_id and objective FIRST so a mention of another function in
                # acceptance_criteria (e.g. "generate_dispatch_report calls simulate_dispatch")
                # never overwrites the task's actual symbol!
                matched_sym: str | None = None
                primary_text = f"{t.task_id} {t.objective}"
                for sym_spec in fb.symbols:
                    short_name = sym_spec.target_symbol.split(".")[-1]
                    if sym_spec.target_symbol in primary_text or re.search(rf"\b{re.escape(short_name)}\b", primary_text):
                        matched_sym = sym_spec.target_symbol
                        break
                if matched_sym is None:
                    secondary_text = f"{' '.join(t.acceptance_criteria)} {' '.join(t.test_assertions)}"
                    for sym_spec in fb.symbols:
                        short_name = sym_spec.target_symbol.split(".")[-1]
                        if sym_spec.target_symbol in secondary_text or re.search(rf"\b{re.escape(short_name)}\b", secondary_text):
                            matched_sym = sym_spec.target_symbol
                            break
                if matched_sym is not None:
                    t.target_symbol = matched_sym

        new_tasks.append(t)

    if expanded_any:
        batch.tasks = new_tasks
        batch.execution_order = [t.task_id for t in new_tasks]


def _prepare_batch_and_workspace(
    batch: TaskBatch,
    existing_snapshots: dict[str, str],
    plan: Plan | None = None,
) -> tuple[TaskBatch, dict[str, str]]:
    """
    Normalize batch relevant_files and execution_order, topologically sort tasks,
    pre-create missing files and Python package `__init__.py` markers in the active
    workspace, infer missing `target_symbol`s, and snapshot pre-batch file contents.
    """
    from .tools import create_file, read_file

    ws_root = get_workspace_root()
    union_files = sorted(
        {
            _normalize_workspace_path(path)
            for task in batch.tasks
            for path in task.relevant_files
            if _normalize_workspace_path(path) not in ("", ".", "..")
        }
        | {
            _normalize_workspace_path(path)
            for path in batch.relevant_files
            if _normalize_workspace_path(path) not in ("", ".", "..")
        }
    )
    batch.relevant_files = union_files
    batch.test_assertions = [
        a.strip() for a in batch.test_assertions if a and a.strip()
    ]

    for task in batch.tasks:
        task.relevant_files = [
            _normalize_workspace_path(p)
            for p in (task.relevant_files or union_files)
            if _normalize_workspace_path(p) not in ("", ".", "..")
        ]
        task.test_assertions = [
            a.strip() for a in task.test_assertions if a and a.strip()
        ]

    _align_batch_with_plan_blueprints(batch, plan)

    if len(batch.tasks) > 1:
        filtered_tasks = [
            t for t in batch.tasks if not _is_redundant_empty_file_task(t)
        ]
        if filtered_tasks:
            batch.tasks = filtered_tasks

    file_counts: dict[str, int] = {}
    for task in batch.tasks:
        for f in task.relevant_files:
            file_counts[f] = file_counts.get(f, 0) + 1

    class_prefix_by_file: dict[str, str] = {}
    for task in batch.tasks:
        if (
            task.relevant_files
            and task.relevant_files[0].endswith(".py")
            and task.target_symbol
            and "." in task.target_symbol
        ):
            class_prefix_by_file[task.relevant_files[0]] = task.target_symbol.split(".", 1)[0]

    for task in batch.tasks:
        primary = task.relevant_files[0] if task.relevant_files else ""
        if not primary.endswith(".py"):
            task.target_symbol = None
            task.test_assertions = []
            continue

        inferred = _infer_task_symbol(task, file_counts.get(primary, 0) > 1)
        if inferred:
            if (
                "." not in inferred
                and primary in class_prefix_by_file
                and inferred != class_prefix_by_file[primary]
            ):
                inferred = f"{class_prefix_by_file[primary]}.{inferred}"
            task.target_symbol = inferred

        if (
            task.target_symbol is None
            and len(task.relevant_files) == 1
            and Path(task.relevant_files[0]).name in ("main.py", "app.py", "hello.py")
        ):
            task.test_assertions = []

    valid_ids = {t.task_id for t in batch.tasks}
    for task in batch.tasks:
        task.dependencies = [
            dep for dep in task.dependencies if dep in valid_ids and dep != task.task_id
        ]
    ordered_ids = [tid for tid in batch.execution_order if tid in valid_ids]
    for task in batch.tasks:
        if task.task_id not in ordered_ids:
            ordered_ids.append(task.task_id)

    batch.execution_order = _sort_execution_order_topologically(
        batch.tasks, ordered_ids
    )

    # Only create `__init__.py` markers for directories that contain `.py` modules
    package_init_files: set[str] = set()
    for rel_path in list(batch.relevant_files):
        if not rel_path.endswith(".py"):
            continue
        parts = Path(rel_path).parts
        if len(parts) > 1:
            for depth in range(1, len(parts)):
                pkg_dir = Path(*parts[:depth])
                if pkg_dir.parts[0] not in (".staging", ".checkpoint", ".git", "__pycache__"):
                    package_init_files.add(str(pkg_dir / "__init__.py"))

    snapshots = dict(existing_snapshots)

    print(f"\n=== FOREMAN REPOSITORY PREPARATION ({ws_root}) ===")
    for init_rel in sorted(package_init_files):
        physical_init = _sandbox_repo_path(init_rel)
        try:
            res = read_file.func(path=physical_init)
            snapshots.setdefault(init_rel, res.get("content", ""))
        except FileNotFoundError:
            create_file.func(path=physical_init, content="")
            snapshots.setdefault(init_rel, "")
            print(f"Created package marker {ws_root.name}/{physical_init}")
        except Exception:
            pass

    for path in batch.relevant_files:
        physical_path = _sandbox_repo_path(path)
        try:
            res = read_file.func(path=physical_path)
            snapshots.setdefault(path, res.get("content", ""))
            print(f"{ws_root.name}/{physical_path} already exists")
        except FileNotFoundError:
            create_file.func(path=physical_path, content="")
            snapshots.setdefault(path, "")
            print(f"Created {ws_root.name}/{physical_path}")
        except Exception as exc:
            print(f"Warning: could not read {physical_path}: {exc}")
            try:
                create_file.func(path=physical_path, content="")
                snapshots.setdefault(path, "")
                print(f"Created {ws_root.name}/{physical_path} (after read error)")
            except Exception as create_exc:
                print(f"Error creating {physical_path}: {create_exc}")

    return batch, snapshots


def _build_effective_validation_batch(task_batches: list[TaskBatch]) -> TaskBatch:
    """
    Combine the initial TaskBatch with any updates from subsequent corrective
    TaskBatches so original `test_assertions` are preserved when corrective
    tasks omit them, or updated when corrective tasks refine them.
    """
    primary = task_batches[0].model_copy(deep=True)
    if len(task_batches) == 1:
        return primary

    tasks_by_id: dict[str, Task] = {t.task_id: t for t in primary.tasks}
    tasks_by_sym: dict[tuple[str, str], Task] = {}
    for t in primary.tasks:
        if t.target_symbol and t.relevant_files:
            tasks_by_sym[(t.relevant_files[0], t.target_symbol)] = t

    union_files = set(primary.relevant_files)
    for corr_batch in task_batches[1:]:
        union_files.update(corr_batch.relevant_files)
        if corr_batch.test_assertions:
            primary.test_assertions = list(corr_batch.test_assertions)
        for ct in corr_batch.tasks:
            matched: Task | None = tasks_by_id.get(ct.task_id)
            if matched is None and ct.target_symbol and ct.relevant_files:
                matched = tasks_by_sym.get((ct.relevant_files[0], ct.target_symbol))
            if matched is not None:
                if ct.test_assertions:
                    matched.test_assertions = list(ct.test_assertions)
                if ct.acceptance_criteria:
                    matched.acceptance_criteria = list(ct.acceptance_criteria)

    primary.relevant_files = sorted(union_files)
    return primary


def _fence_lang_for_path(path: str) -> str:
    ext = Path(path).suffix.lower().lstrip(".")
    return {"py": "python", "yml": "yaml", "md": "markdown"}.get(ext, ext or "text")


def _gather_repository_context(batch: TaskBatch, task_results: list[TaskResult]) -> str:
    """Inspect the active workspace repository files and construct textual evidence."""
    from .tools import read_file

    relevant_files = sorted(
        {_normalize_workspace_path(p) for p in batch.relevant_files}
        | {
            _normalize_workspace_path(file_path)
            for result in task_results
            for file_path in result.files_changed
        }
    )

    repository_evidence: list[str] = []
    for path in relevant_files:
        physical_path = _sandbox_repo_path(path)
        lang = _fence_lang_for_path(path)
        try:
            result = read_file.func(path=physical_path)
            content = result.get("content", "")
            content_display = content if content.strip() else "[EMPTY FILE - 0 bytes]"
            repository_evidence.append(f"FILE: {path}\n~~~{lang}\n{content_display}\n~~~")
        except Exception as exc:
            repository_evidence.append(
                f"FILE: {path}\nREAD_ERROR: {type(exc).__name__}: {exc}"
            )

    return "\n\n".join(repository_evidence)


def _gather_full_repository_context() -> str:
    """Gather context from all user files in the active workspace for Planner review."""
    from .tools import read_file

    ws_root = get_workspace_root()
    repository_evidence: list[str] = []
    for fpath in list_workspace_files():
        try:
            relative_path = fpath.relative_to(ws_root)
            logical_path = str(relative_path)
            lang = _fence_lang_for_path(logical_path)
            result = read_file.func(path=logical_path)
            content = result.get("content", "")
            if not content.strip() and fpath.name == "__init__.py":
                content_display = "[PACKAGE __init__.py - 0 bytes]"
            else:
                content_display = content if content.strip() else "[EMPTY FILE - 0 bytes]"
            repository_evidence.append(
                f"FILE: {logical_path}\n~~~{lang}\n{content_display}\n~~~"
            )
        except Exception as exc:
            repository_evidence.append(
                f"FILE: {logical_path}\nREAD_ERROR: {type(exc).__name__}: {exc}"
            )

    return (
        "\n\n".join(repository_evidence)
        if repository_evidence
        else f"[No files in {ws_root.name}/]"
    )


def _persist_node_state(state: WorkflowState, updates: dict) -> dict:
    """Merge node state updates and save checkpoint to `<workspace>/.checkpoint/state.json`."""
    merged: WorkflowState = {**state, **updates}  # type: ignore[assignment]
    save_checkpoint(merged)
    return updates


def build_planner() -> PlannerAgent:
    return PlannerAgent()


def build_foreman() -> ForemanAgent:
    return ForemanAgent()


def build_worker() -> AiderHarness:
    return AiderHarness()


# ============================================================================
# 1. Planner — Blueprinting
# ============================================================================


def planner_node(state: WorkflowState) -> dict:
    planner = build_planner()
    enriched_arch = _enrich_architecture_context(state["architecture_context"])
    ledger_ctx = format_ledger_for_prompt(state["ledger"], state["previous_cycle"])

    plan = planner.run(
        project_goal=state["project_goal"],
        architecture_context=enriched_arch,
        worker_profile=state["worker_profile"],
        previous_cycle=state["previous_cycle"],
        ledger_context=ledger_ctx,
    )

    if not isinstance(plan, Plan):
        raise TypeError("PlannerAgent.run() must return Plan.")

    print("\n=== PLANNER RESULT ===")
    print(plan.model_dump_json(indent=2))

    updated_ledger = record_ledger_entry(
        state["ledger"],
        cycle_id=state["cycle_id"],
        layer="planner",
        action="create_plan",
        target=", ".join(plan.affected_areas) or "workspace",
        status="success",
        detail=f"Planned {len(plan.acceptance_criteria)} acceptance criteria: {plan.objective}",
    )

    return _persist_node_state(
        state,
        {
            "plan": plan,
            "ledger": updated_ledger,
            "phase": "planned",
        },
    )


# ============================================================================
# 2. Foreman — Task Atomization & Preparation
# ============================================================================


def foreman_node(state: WorkflowState) -> dict:
    plan = state["plan"]
    if plan is None:
        raise RuntimeError("Planner output missing.")

    foreman = build_foreman()
    enriched_arch = _enrich_architecture_context(state["architecture_context"])
    ledger_ctx = format_ledger_for_prompt(state["ledger"], state["previous_cycle"])

    batch = foreman.create_batch(
        project_goal=state["project_goal"],
        architecture_context=enriched_arch,
        worker_profile=state["worker_profile"],
        previous_cycle=state["previous_cycle"],
        plan=plan,
        ledger_context=ledger_ctx,
    )

    if not isinstance(batch, TaskBatch):
        raise TypeError("ForemanAgent.create_batch() must return TaskBatch.")

    batch, snapshots = _prepare_batch_and_workspace(
        batch, state.get("file_snapshots", {}), plan=plan
    )

    print("\n=== FOREMAN RESULT ===")
    print(batch.model_dump_json(indent=2))

    task_desc = ", ".join(
        f"{t.task_id}({t.target_symbol or ','.join(t.relevant_files)})"
        for t in batch.tasks
    )
    updated_ledger = record_ledger_entry(
        state["ledger"],
        cycle_id=state["cycle_id"],
        layer="foreman",
        action="atomize_tasks",
        target=batch.batch_id,
        status="success",
        detail=f"Atomized plan into {len(batch.tasks)} task(s): {task_desc}",
    )

    return _persist_node_state(
        state,
        {
            "task_batches": [batch],
            "file_snapshots": snapshots,
            "ledger": updated_ledger,
            "phase": "task_ready",
        },
    )


# ============================================================================
# 3. Worker — Atomic Execution (Direct or Staged, with Intra-Batch Checkpointing)
# ============================================================================


def worker_node(state: WorkflowState) -> dict:
    if not state["task_batches"]:
        raise RuntimeError("No TaskBatch received.")

    ws_root = get_workspace_root()
    batch = state["task_batches"][-1]
    worker = build_worker()

    print("\n=== WORKER SESSION START ===")
    print(f"Batch: {batch.batch_id}")

    verified_by_id = {
        r.task_id: r
        for r in state.get("task_results", [])
        if r.status == "completed"
    }
    merged_by_id = {r.task_id: r for r in state.get("task_results", [])}
    attempts = dict(state.get("worker_attempts", {}))
    staged_files: dict[str, dict[str, str]] = {
        _normalize_workspace_path(k): dict(v)
        for k, v in state.get("staged_files", {}).items()
    }
    updated_ledger = list(state.get("ledger", []))

    def _on_single_task_complete(
        task_obj: Task,
        result: TaskResult,
        live_staged: dict[str, dict[str, str]],
    ) -> None:
        nonlocal updated_ledger, staged_files
        attempts[result.task_id] = attempts.get(result.task_id, 0) + 1
        merged_by_id[result.task_id] = result
        staged_files = {
            _normalize_workspace_path(k): dict(v) for k, v in live_staged.items()
        }

        target_file = (
            _normalize_workspace_path(task_obj.relevant_files[0])
            if task_obj.relevant_files
            else "unknown"
        )
        symbol_label = (
            f"{target_file}::{task_obj.target_symbol}"
            if task_obj.target_symbol
            else target_file
        )
        updated_ledger = record_ledger_entry(
            updated_ledger,
            cycle_id=state["cycle_id"],
            layer="worker",
            action="execute_task",
            target=f"{result.task_id} ({symbol_label})",
            status="success" if result.status == "completed" else "warning",
            detail=result.summary,
        )

        partial_state: WorkflowState = {
            **state,
            "task_results": list(merged_by_id.values()),
            "worker_attempts": dict(attempts),
            "staged_files": dict(staged_files),
            "ledger": list(updated_ledger),
        }
        save_checkpoint(partial_state)

    results = worker.execute(
        batch,
        existing_staged=staged_files,
        file_snapshots=state.get("file_snapshots", {}),
        project_goal=state["project_goal"],
        verified_task_results=verified_by_id,
        on_task_complete=_on_single_task_complete,
        plan=state.get("plan"),
    )

    print("\n=== WORKER SESSION COMPLETE ===")

    tasks_by_id = {t.task_id: t for t in batch.tasks}
    for result in results:
        if result.task_id not in merged_by_id:
            merged_by_id[result.task_id] = result
            attempts[result.task_id] = attempts.get(result.task_id, 0) + 1
            task_obj = tasks_by_id.get(result.task_id)
            target_file = (
                _normalize_workspace_path(task_obj.relevant_files[0])
                if task_obj and task_obj.relevant_files
                else "unknown"
            )
            if result.staged_file:
                staged_path = ws_root / result.staged_file
                if staged_path.is_file():
                    raw_snippet = staged_path.read_text(encoding="utf-8").strip()
                    snippet = ensure_class_wrapper(
                        raw_snippet,
                        task_obj.target_symbol if task_obj else None,
                    ).strip()
                    if snippet:
                        staged_files[target_file] = upsert_staged_snippet(
                            staged_files.get(target_file, {}),
                            result.task_id,
                            snippet,
                            target_symbol=task_obj.target_symbol if task_obj else None,
                        )

    return _persist_node_state(
        state,
        {
            "task_results": list(merged_by_id.values()),
            "worker_attempts": attempts,
            "staged_files": staged_files,
            "ledger": updated_ledger,
            "phase": "worker_complete",
        },
    )


# ============================================================================
# 4. Foreman Assembler — Stitch Staged Functions & Class Methods into Modules
# ============================================================================


def foreman_assemble_node(state: WorkflowState) -> dict:
    """
    When a Python file was split into multiple function-level or class-method-level
    tasks in `.staging/`, Foreman assembles the individual snippets into the
    final target module, verifying via AST that every function, class, and
    method in `ast_merged` is retained.
    """
    from .tools import write_file

    staged_files = state.get("staged_files", {})
    if not staged_files:
        return _persist_node_state(state, {"phase": "assembled"})

    ws_root = get_workspace_root()
    foreman = build_foreman()
    plan = state["plan"]
    snapshots = state.get("file_snapshots", {})
    updated_ledger = list(state["ledger"])
    ledger_ctx = format_ledger_for_prompt(updated_ledger, state["previous_cycle"])

    print("\n=== FOREMAN MODULE ASSEMBLY ===")

    for target_file, task_snippets in sorted(staged_files.items()):
        if not task_snippets:
            continue

        norm_target = _normalize_workspace_path(target_file)
        if not norm_target.endswith(".py"):
            continue

        ast_merged = merge_staged_snippets_ast(
            target_filename=norm_target,
            base_source=snapshots.get(norm_target, ""),
            ordered_snippets=list(task_snippets.items()),
        )
        expected_sigs = extract_python_signatures(ast_merged)
        expected_symbols = extract_all_ast_symbol_names(ast_merged)

        final_code = ast_merged
        assembly_mode = "AST-deterministic"

        if len(task_snippets) > 1 and plan is not None:
            try:
                assembled = foreman.assemble_module(
                    target_file=norm_target,
                    project_goal=state["project_goal"],
                    plan=plan,
                    staged_snippets=task_snippets,
                    ast_merged_draft=ast_merged,
                    ledger_context=ledger_ctx,
                )
                candidate = assembled.assembled_code.strip()
                if candidate:
                    ast.parse(candidate)
                    candidate_symbols = extract_all_ast_symbol_names(candidate)
                    missing_symbols = sorted(expected_symbols - candidate_symbols)
                    if not missing_symbols:
                        final_code = candidate
                        assembly_mode = f"Foreman-LLM + AST verified ({assembled.notes or 'clean'})"
                    else:
                        print(
                            f"Foreman assembly omitted {missing_symbols}; "
                            f"retaining complete AST merge for {norm_target}."
                        )
            except Exception as exc:
                print(
                    f"Foreman LLM assembly fallback to AST merge for {norm_target}: {exc}"
                )

        physical_path = _sandbox_repo_path(norm_target)
        write_file.func(path=physical_path, content=final_code)
        print(
            f"Assembled {ws_root.name}/{physical_path} from {len(task_snippets)} staged snippet(s) "
            f"[{assembly_mode}]"
        )

        updated_ledger = record_ledger_entry(
            updated_ledger,
            cycle_id=state["cycle_id"],
            layer="assembler",
            action="assemble_module",
            target=norm_target,
            status="success",
            detail=(
                f"Assembled {len(task_snippets)} snippet(s) "
                f"({', '.join(task_snippets.keys())}) -> { '; '.join(expected_sigs)}"
            ),
        )

    return _persist_node_state(
        state,
        {
            "ledger": updated_ledger,
            "phase": "assembled",
        },
    )


# ============================================================================
# 5. Validator
# ============================================================================


def validator_node(state: WorkflowState) -> dict:
    if not state["task_batches"]:
        raise RuntimeError("No TaskBatch available for validation.")

    effective_batch = _build_effective_validation_batch(state["task_batches"])
    validation = validate_batch(
        batch=effective_batch,
        task_results=state["task_results"],
    )

    if not isinstance(validation, ValidationResult):
        raise TypeError("validate_batch() must return ValidationResult.")

    print("\n=== VALIDATOR RESULT ===")
    print(validation.model_dump_json(indent=2))

    detail = (
        f"All {len(validation.checks)} checks passed."
        if validation.passed
        else f"Failed ({', '.join(validation.failed_task_ids) or 'batch'}): {'; '.join(validation.errors[:2])}"
    )
    updated_ledger = record_ledger_entry(
        state["ledger"],
        cycle_id=state["cycle_id"],
        layer="validator",
        action="validate_batch",
        target=", ".join(validation.files_checked),
        status="success" if validation.passed else "failure",
        detail=detail,
    )

    return _persist_node_state(
        state,
        {
            "validation_result": validation,
            "ledger": updated_ledger,
            "phase": "validated",
        },
    )


# ============================================================================
# 6. Foreman — Review, Self-Repair & Corrective Rebuild
# ============================================================================


def foreman_review_node(state: WorkflowState) -> dict:
    if not state["task_batches"]:
        raise RuntimeError("No TaskBatch available for Foreman review.")
    if not state["task_results"]:
        raise RuntimeError("No TaskResults available for Foreman review.")

    validation = state["validation_result"]
    if validation is None:
        raise RuntimeError("ValidationResult missing before Foreman review.")

    plan = state["plan"]
    if plan is None:
        raise RuntimeError("Planner output missing before Foreman review.")

    batch = _build_effective_validation_batch(state["task_batches"])
    repository_context = _gather_repository_context(
        batch, state["task_results"]
    )
    enriched_arch = _enrich_architecture_context(state["architecture_context"])
    ledger_ctx = format_ledger_for_prompt(state["ledger"], state["previous_cycle"])

    foreman = build_foreman()
    review = foreman.review(
        project_goal=state["project_goal"],
        architecture_context=enriched_arch,
        plan=plan,
        batch=batch,
        task_results=state["task_results"],
        validation_result=validation,
        repository_context=repository_context,
        worker_attempts=state["worker_attempts"],
        foreman_repair_attempts=state["foreman_repair_attempts"],
        max_foreman_repairs=MAX_FOREMAN_REPAIRS,
        ledger_context=ledger_ctx,
    )

    if not isinstance(review, ForemanReview):
        raise TypeError("ForemanAgent.review() must return ForemanReview.")

    print("\n=== FOREMAN REVIEW ===")
    print(review.model_dump_json(indent=2))

    updated_ledger = record_ledger_entry(
        state["ledger"],
        cycle_id=state["cycle_id"],
        layer="foreman",
        action=f"review_{review.decision}",
        target=batch.batch_id,
        status="success" if review.decision == "accept" else "warning",
        detail=review.summary,
    )

    return _persist_node_state(
        state,
        {
            "foreman_review": review,
            "ledger": updated_ledger,
            "phase": f"foreman_review_{review.decision}",
        },
    )


def foreman_self_repair_node(state: WorkflowState) -> dict:
    """
    Foreman executes direct localized repairs with:
    1. Automatic markdown-fence sanitization (via `write_file`/`edit_file`).
    2. Immediate syntax verification (`_is_valid_workspace_file`) — if a Foreman patch
       introduces a syntax error, it is immediately reverted to `before_content` so
       Foreman can NEVER corrupt a file on disk.
    3. Zero-Diff Stagnation Detection.
    """
    from .checkpoint import _is_valid_workspace_file
    from .tools import edit_file, read_file, write_file

    ws_root = get_workspace_root()
    review = state["foreman_review"]
    print("\n=== FOREMAN DIRECT SELF-REPAIR ===")

    repaired_files: list[str] = []
    unchanged_files: list[str] = []
    repair_errors: list[str] = []

    for repair in review.direct_repairs:
        physical_path = _sandbox_repo_path(repair.path)
        norm_rel = _normalize_workspace_path(repair.path)
        abs_target = ws_root / norm_rel
        print(f"Applying Foreman self-repair to {ws_root.name}/{physical_path} ({repair.action})")
        try:
            before_content = ""
            try:
                before_content = read_file.func(path=physical_path).get("content", "")
            except Exception:
                before_content = ""

            if repair.action == "edit":
                if not repair.target_text:
                    raise ValueError(
                        "Foreman 'edit' repair requires non-empty target_text."
                    )
                edit_file.func(
                    path=physical_path,
                    old_text=repair.target_text,
                    new_text=repair.replacement_text,
                )
            else:
                if not repair.content.strip():
                    raise ValueError(
                        "Foreman 'overwrite' repair requires non-empty content."
                    )
                write_file.func(
                    path=physical_path,
                    content=repair.content,
                )

            if not _is_valid_workspace_file(abs_target):
                abs_target.write_text(before_content, encoding="utf-8")
                raise ValueError(
                    f"Foreman patch introduced invalid syntax in {norm_rel}; reverted to previous content."
                )

            after_content = read_file.func(path=physical_path).get("content", "")
            if before_content.strip() == after_content.strip():
                print(
                    f"[STAGNATION GUARD] Foreman self-repair on {ws_root.name}/{physical_path} "
                    f"produced zero content change on disk."
                )
                unchanged_files.append(norm_rel)
            else:
                repaired_files.append(norm_rel)
        except Exception as exc:
            err_msg = f"Foreman self-repair failed on {repair.path}: {type(exc).__name__}: {exc}"
            print(f"WARNING: {err_msg}")
            repair_errors.append(err_msg)

    status = "completed" if repaired_files and not repair_errors else "no_change"
    if repair_errors:
        status = "failure"

    repair_id = f"foreman_self_repair_{state['foreman_repair_attempts'] + 1}"
    summary_text = (
        f"Foreman direct repair: {', '.join(review.repair_instructions or ['self-repaired'])}"
        if repaired_files
        else f"Foreman self-repair produced no valid changes on {unchanged_files or [r.path for r in review.direct_repairs]}"
    )
    repair_task_result = TaskResult(
        task_id=repair_id,
        status=status,
        summary=summary_text,
        files_changed=repaired_files,
        errors=repair_errors,
    )

    next_repair_attempts = (
        MAX_FOREMAN_REPAIRS
        if not repaired_files
        else state["foreman_repair_attempts"] + 1
    )

    updated_ledger = record_ledger_entry(
        state["ledger"],
        cycle_id=state["cycle_id"],
        layer="foreman",
        action="self_repair",
        target=", ".join(repaired_files or unchanged_files) or "none",
        status="success" if status == "completed" else "warning",
        detail=repair_task_result.summary,
    )

    return _persist_node_state(
        state,
        {
            "task_results": state["task_results"] + [repair_task_result],
            "foreman_repair_attempts": next_repair_attempts,
            "ledger": updated_ledger,
            "phase": "foreman_repaired",
        },
    )


def foreman_corrective_node(state: WorkflowState) -> dict:
    """
    Foreman rolls back corrupted files and creates a surgical corrective TaskBatch:
    - Strictly filters out tasks targeting files that already passed validation so
      working files (like `config/dam_profile.json` or `data/telemetry.csv`) are NEVER
      re-edited and broken during a rebuild of another file.
    - Resets any syntax-corrupted target file on disk before Aider runs so Aider never
      stalls with `no_change` on malformed file content.
    """
    from .checkpoint import _is_valid_workspace_file
    from .tools import write_file

    ws_root = get_workspace_root()
    foreman = build_foreman()
    batch = _build_effective_validation_batch(state["task_batches"])
    review = state["foreman_review"]
    validation = state.get("validation_result")
    snapshots = state.get("file_snapshots", {})
    ledger_ctx = format_ledger_for_prompt(state["ledger"], state["previous_cycle"])

    if review and review.rollback_files:
        print("\n=== FOREMAN ROLLING BACK CORRUPTED FILES ===")
        for path in review.rollback_files:
            logical = _normalize_workspace_path(path)
            if logical in snapshots:
                physical_path = _sandbox_repo_path(logical)
                write_file.func(path=physical_path, content=snapshots[logical])
                print(f"Reverted {ws_root.name}/{physical_path} to pre-batch snapshot")

    print("\n=== FOREMAN GENERATING CORRECTIVE BATCH ===")
    corrective_batch = foreman.create_corrective_batch(
        batch=batch,
        review=review,
        task_results=state["task_results"],
        validation_result=validation,
        ledger_context=ledger_ctx,
    )

    # Strictly filter corrective tasks so already-passing tasks/files are NEVER re-run
    if validation and validation.failed_task_ids:
        failed_ids = set(validation.failed_task_ids)
        orig_tasks_by_id = {t.task_id: t for t in batch.tasks}
        failed_files = {
            _normalize_workspace_path(rf)
            for fid in failed_ids
            if fid in orig_tasks_by_id
            for rf in orig_tasks_by_id[fid].relevant_files
        }
        filtered_corr_tasks = [
            ct for ct in corrective_batch.tasks
            if ct.task_id in failed_ids
            or any(_normalize_workspace_path(rf) in failed_files for rf in ct.relevant_files)
        ]
        if filtered_corr_tasks:
            corrective_batch.tasks = filtered_corr_tasks
        else:
            # Fallback: copy the exact failed Task definitions from the effective batch
            corrective_batch.tasks = [
                orig_tasks_by_id[fid].model_copy(deep=True)
                for fid in validation.failed_task_ids
                if fid in orig_tasks_by_id
            ]
        corrective_batch.relevant_files = sorted(
            {
                _normalize_workspace_path(rf)
                for ct in corrective_batch.tasks
                for rf in ct.relevant_files
            }
        )
        corrective_batch.execution_order = [ct.task_id for ct in corrective_batch.tasks]

    # If any file targeted by the corrective batch is currently syntax-corrupted on disk,
    # reset it to its pre-batch snapshot (or empty string) so Aider can edit it cleanly.
    for rel_f in corrective_batch.relevant_files:
        norm_f = _normalize_workspace_path(rel_f)
        abs_f = ws_root / norm_f
        if abs_f.is_file() and not _is_valid_workspace_file(abs_f):
            fallback_src = snapshots.get(norm_f, "")
            abs_f.write_text(fallback_src, encoding="utf-8")
            print(f"[RECOVERY] Cleared corrupted syntax in {ws_root.name}/{norm_f} before Worker rebuild")

    corrective_batch, updated_snapshots = _prepare_batch_and_workspace(
        corrective_batch, snapshots, plan=None
    )

    rebuild_ids = {t.task_id for t in corrective_batch.tasks}
    retained_results = [
        r for r in state["task_results"] if r.task_id not in rebuild_ids
    ]

    updated_ledger = record_ledger_entry(
        state["ledger"],
        cycle_id=state["cycle_id"],
        layer="foreman",
        action="corrective_batch",
        target=corrective_batch.batch_id,
        status="warning",
        detail=f"Created surgical corrective batch for failed task(s): {', '.join(t.task_id for t in corrective_batch.tasks)}",
    )

    return _persist_node_state(
        state,
        {
            "task_batches": state["task_batches"] + [corrective_batch],
            "task_results": retained_results,
            "file_snapshots": updated_snapshots,
            "rebuild_attempts": state["rebuild_attempts"] + 1,
            "ledger": updated_ledger,
            "phase": "corrective_task_ready",
        },
    )


# ============================================================================
# 7. Planner — Acceptance Review
# ============================================================================


def planner_review_node(state: WorkflowState) -> dict:
    """Planner performs the final evaluation against the initial blueprint."""
    validation = state.get("validation_result")
    plan = state["plan"]

    if validation is not None and not validation.passed:
        print("\n=== PLANNER FINAL ACCEPTANCE REVIEW (FAST-REJECT: VALIDATION FAILED) ===")
        review = PlannerReview(
            criteria_evaluation=[f"FAIL: {err}" for err in validation.errors[:5]],
            feedback=list(validation.errors[:5]),
            summary=f"Cycle rejected deterministically because {len(validation.errors)} validation error(s) remain.",
            accepted=False,
        )
        print(review.model_dump_json(indent=2))
        updated_ledger = record_ledger_entry(
            state["ledger"],
            cycle_id=state["cycle_id"],
            layer="planner",
            action="final_review",
            target="project",
            status="failure",
            detail=review.summary,
        )
        return _persist_node_state(
            state,
            {
                "planner_review": review,
                "ledger": updated_ledger,
                "phase": "planner_revision_needed",
            },
        )

    planner = build_planner()
    repository_context = _gather_full_repository_context()
    ledger_ctx = format_ledger_for_prompt(state["ledger"], state["previous_cycle"])

    print("\n=== PLANNER FINAL ACCEPTANCE REVIEW ===")
    foreman_rev = state.get("foreman_review")
    try:
        review = planner.review(
            project_goal=state["project_goal"],
            plan=plan,
            repository_context=repository_context,
            validation_result=validation,
            ledger_context=ledger_ctx,
        )
    except Exception as exc:
        if validation is not None and validation.passed and (foreman_rev is None or foreman_rev.accepted):
            print(
                f"[WARN] Planner LLM review raised {type(exc).__name__} after all "
                f"{len(validation.checks)} deterministic checks and Foreman review passed. "
                f"Auto-accepting verified build."
            )
            review = PlannerReview(
                criteria_evaluation=[f"PASS: {c}" for c in (plan.acceptance_criteria if plan else [])],
                feedback=[],
                summary="Accepted via deterministic validator and Foreman verification.",
                accepted=True,
            )
        else:
            raise

    print(review.model_dump_json(indent=2))

    updated_ledger = record_ledger_entry(
        state["ledger"],
        cycle_id=state["cycle_id"],
        layer="planner",
        action="final_review",
        target="project",
        status="success" if review.accepted else "failure",
        detail=review.summary,
    )

    return _persist_node_state(
        state,
        {
            "planner_review": review,
            "ledger": updated_ledger,
            "phase": "planner_accepted" if review.accepted else "planner_revision_needed",
        },
    )


# ============================================================================
# Routing, Stagnation Circuit Breaker & Graph Construction
# ============================================================================


def _count_consecutive_identical_validator_failures(state: WorkflowState) -> int:
    """
    Count how many times the current validation error signature (or any individual
    failed task/error in `validation_result`) has repeated across validation runs in
    this cycle — catching both consecutive repeats (`A -> A`) and oscillations (`A -> B -> A`).
    """
    validator_entries = [
        e for e in state.get("ledger", [])
        if e.layer == "validator" and e.action == "validate_batch"
    ]
    if not validator_entries or validator_entries[-1].status != "failure":
        return 0

    latest_detail = validator_entries[-1].detail
    exact_matches = sum(
        1 for e in validator_entries if e.status == "failure" and e.detail == latest_detail
    )

    # Also check if the primary error message (first error in validation.errors) has
    # appeared in previous validator ledger entries even when secondary errors toggled.
    validation = state.get("validation_result")
    primary_err_matches = 0
    if validation and validation.errors:
        first_err_prefix = validation.errors[0].splitlines()[0][:80]
        if first_err_prefix:
            primary_err_matches = sum(
                1 for e in validator_entries
                if e.status == "failure" and first_err_prefix in e.detail
            )

    return max(exact_matches, primary_err_matches)


def route_from_entry(state: WorkflowState) -> str:
    """Route from START to the appropriate node based on the reconciled checkpoint phase."""
    phase = state.get("phase", "initial")

    if phase == "initial" or state.get("plan") is None:
        return "planner"
    if phase == "planned" or not state.get("task_batches"):
        return "foreman"
    if phase in ("task_ready", "corrective_task_ready"):
        return "worker"
    if phase == "worker_complete":
        return "foreman_assemble"
    if phase in ("assembled", "foreman_repaired"):
        return "validator"
    if phase == "validated":
        return "foreman_review"
    if phase == "foreman_review_accept":
        return "planner_review"
    if phase == "foreman_review_repair":
        return "foreman_self_repair"
    if phase == "foreman_review_rebuild":
        return "foreman_corrective"

    return "planner"


def route_after_planner(state: WorkflowState) -> str:
    """Allow `--plan-only` runs to checkpoint the Planner blueprint and exit cleanly."""
    if state.get("rework_mode") == "plan_only":
        return "end"
    return "foreman"


def route_after_foreman_review(state: WorkflowState) -> str:
    review = state["foreman_review"]
    if review is None:
        raise RuntimeError("Foreman review missing.")

    validation = state.get("validation_result")
    has_hard_errors = bool(
        validation
        and any(
            err.startswith(
                (
                    "syntax error",
                    "empty file",
                    "missing file",
                    "missing symbol",
                    "execution failed",
                    "assertion failed",
                    "pytest failed",
                    "verification command failed",
                )
            )
            for err in validation.errors
        )
    )

    repeat_failures = _count_consecutive_identical_validator_failures(state)
    last_task_no_change = bool(
        state.get("task_results")
        and state["task_results"][-1].status == "no_change"
    )

    if has_hard_errors and (
        repeat_failures >= 3
        or (repeat_failures >= 2 and last_task_no_change and state["rebuild_attempts"] >= 1)
    ):
        print(
            f"[STAGNATION CIRCUIT BREAKER] Identical validation error repeated "
            f"{repeat_failures} consecutive time(s) without progress. "
            f"Breaking repair/rebuild loop."
        )
        return "accepted"

    if review.decision == "accept":
        if not has_hard_errors:
            return "accepted"
        print(
            "WARNING: Foreman chose 'accept' while deterministic validation/assertion errors remain. "
            "Overriding to rework."
        )
        review.accepted = False
        if validation:
            if not review.problems:
                review.problems = list(validation.errors)
            if not review.rebuild_task_ids and validation.failed_task_ids:
                review.rebuild_task_ids = list(validation.failed_task_ids)
        if (
            review.direct_repairs
            and state["foreman_repair_attempts"] < MAX_FOREMAN_REPAIRS
            and repeat_failures < 2
        ):
            return "self_repair"
        if state["rebuild_attempts"] < MAX_WORKER_REBUILDS:
            return "rebuild"
        return "accepted"

    if review.decision == "repair":
        if (
            review.direct_repairs
            and state["foreman_repair_attempts"] < MAX_FOREMAN_REPAIRS
            and not (state["foreman_repair_attempts"] >= 1 and repeat_failures >= 2)
        ):
            return "self_repair"
        if not review.direct_repairs and not has_hard_errors:
            print("Foreman requested repair with empty direct_repairs and no hard errors; accepting.")
            review.accepted = True
            return "accepted"
        print("Foreman self-repair unavailable, stagnant, or exceeded limit. Escalating to rebuild.")
        if validation and not review.rebuild_task_ids and validation.failed_task_ids:
            review.rebuild_task_ids = list(validation.failed_task_ids)

    if state["rebuild_attempts"] >= MAX_WORKER_REBUILDS:
        print(
            f"Worker rebuild attempts reached limit ({MAX_WORKER_REBUILDS}). "
            "Escalating to Planner review."
        )
        return "accepted"

    return "rebuild"


def build_graph():
    graph = StateGraph(WorkflowState)

    graph.add_node("planner", planner_node)
    graph.add_node("foreman", foreman_node)
    graph.add_node("worker", worker_node)
    graph.add_node("foreman_assemble", foreman_assemble_node)
    graph.add_node("validator", validator_node)
    graph.add_node("foreman_review", foreman_review_node)
    graph.add_node("foreman_self_repair", foreman_self_repair_node)
    graph.add_node("foreman_corrective", foreman_corrective_node)
    graph.add_node("planner_review", planner_review_node)

    graph.set_conditional_entry_point(
        route_from_entry,
        {
            "planner": "planner",
            "foreman": "foreman",
            "worker": "worker",
            "foreman_assemble": "foreman_assemble",
            "validator": "validator",
            "foreman_review": "foreman_review",
            "foreman_self_repair": "foreman_self_repair",
            "foreman_corrective": "foreman_corrective",
            "planner_review": "planner_review",
        },
    )

    graph.add_conditional_edges(
        "planner",
        route_after_planner,
        {
            "foreman": "foreman",
            "end": END,
        },
    )
    graph.add_edge("foreman", "worker")
    graph.add_edge("worker", "foreman_assemble")
    graph.add_edge("foreman_assemble", "validator")
    graph.add_edge("validator", "foreman_review")

    graph.add_conditional_edges(
        "foreman_review",
        route_after_foreman_review,
        {
            "self_repair": "foreman_self_repair",
            "rebuild": "foreman_corrective",
            "accepted": "planner_review",
        },
    )

    graph.add_edge("foreman_self_repair", "validator")
    graph.add_edge("foreman_corrective", "worker")
    graph.add_edge("planner_review", END)

    return graph.compile()


app = build_graph()


def run_cycle(
    *,
    cycle_id: int,
    goal: str,
    architecture: str,
    worker_profile: WorkerProfile,
    previous_cycle: CycleResult | None,
    resume: bool = False,
    workspace: str | Path | None = None,
    plan_only: bool = False,
    emit_tests: bool = False,
    suite_rel_path: str = "tests/test_generated.py",
) -> CycleResult:
    """Execute a single orchestration cycle in the active or specified workspace."""
    if workspace is not None:
        set_active_workspace(workspace)

    staging_dir = get_staging_dir()
    initial_state: WorkflowState | None = None

    if resume:
        ckpt = load_checkpoint()
        if ckpt is not None and ckpt.cycle_id == cycle_id:
            initial_state = reconcile_checkpoint_with_disk(ckpt)
            if goal.strip():
                initial_state["project_goal"] = goal
            if architecture.strip():
                initial_state["architecture_context"] = architecture
            initial_state["worker_profile"] = worker_profile
            initial_state["rework_mode"] = "plan_only" if plan_only else "none"

    if initial_state is None:
        if staging_dir.exists():
            for f in staging_dir.rglob("*.py"):
                f.unlink(missing_ok=True)

        initial_state = {
            "cycle_id": cycle_id,
            "project_goal": goal,
            "architecture_context": architecture,
            "worker_profile": worker_profile,
            "previous_cycle": previous_cycle,
            "plan": None,
            "task_batches": [],
            "task_results": [],
            "ledger": [],
            "staged_files": {},
            "file_snapshots": {},
            "worker_attempts": {},
            "foreman_repair_attempts": 0,
            "rebuild_attempts": 0,
            "validation_result": None,
            "foreman_review": None,
            "planner_review": None,
            "rework_mode": "plan_only" if plan_only else "none",
            "phase": "initial",
        }
        save_checkpoint(initial_state)

    final_state = app.invoke(initial_state)

    if plan_only and final_state.get("phase") == "planned":
        return CycleResult(
            cycle_id=cycle_id,
            plan=final_state["plan"],
            task_batches=[],
            task_results=[],
            ledger=final_state["ledger"],
            planner_review=None,
            outcome="planned",
        )

    foreman_review = final_state.get("foreman_review")
    planner_review = final_state.get("planner_review")

    success = (
        planner_review is not None
        and planner_review.accepted
        and (foreman_review is None or foreman_review.accepted)
    )

    if success and emit_tests and final_state.get("task_batches"):
        effective_batch = _build_effective_validation_batch(final_state["task_batches"])
        emit_pytest_suite(effective_batch, suite_rel_path=suite_rel_path)

    outcome = "overall_success" if success else "partial"

    return CycleResult(
        cycle_id=cycle_id,
        plan=final_state["plan"],
        task_batches=final_state["task_batches"],
        task_results=final_state["task_results"],
        ledger=final_state["ledger"],
        planner_review=planner_review,
        outcome=outcome,
    )


def run_project(
    *,
    goal: str,
    architecture: str,
    worker_profile: WorkerProfile,
    max_cycles: int = 3,
    resume: bool = False,
    workspace: str | Path | None = None,
    plan_only: bool = False,
    emit_tests: bool = False,
) -> list[CycleResult]:
    """Execute the outer Planner loop across multiple cycles in the target workspace."""
    if workspace is not None:
        set_active_workspace(workspace)

    history: list[CycleResult] = []
    previous_cycle: CycleResult | None = None
    start_cycle = 1

    if resume:
        ckpt = load_checkpoint()
        if ckpt is not None:
            if ckpt.stages:
                return run_staged_project(
                    goal=goal or ckpt.project_goal,
                    architecture=architecture or ckpt.architecture_context,
                    worker_profile=worker_profile,
                    stages=ckpt.stages,
                    max_cycles_per_stage=max_cycles,
                    resume=True,
                    workspace=workspace,
                    plan_only=plan_only,
                    emit_tests=emit_tests,
                )
            history = list(ckpt.completed_cycles)
            start_cycle = ckpt.cycle_id
            previous_cycle = ckpt.previous_cycle
            print(
                f"\n=== RESUMING PROJECT FROM CHECKPOINT (Workspace: {get_workspace_root()}, "
                f"Cycle {start_cycle}, Phase: {ckpt.phase}) ==="
            )
    else:
        clear_checkpoint()

    for cycle_id in range(start_cycle, max_cycles + 1):
        print(f"\n==========================================")
        print(f"=== STARTING ORCHESTRATION CYCLE {cycle_id}/{max_cycles} ({get_workspace_root()}) ===")
        print(f"==========================================")

        use_resume_this_cycle = resume and (cycle_id == start_cycle)
        result = run_cycle(
            cycle_id=cycle_id,
            goal=goal,
            architecture=architecture,
            worker_profile=worker_profile,
            previous_cycle=previous_cycle,
            resume=use_resume_this_cycle,
            plan_only=plan_only,
            emit_tests=emit_tests,
        )
        history.append(result)

        if plan_only and result.outcome == "planned":
            print("\n=== PLAN-ONLY MODE COMPLETE (Saved to .checkpoint/state.json) ===")
            break

        if result.outcome == "overall_success":
            print(f"\n=== PROJECT ACCEPTED IN CYCLE {cycle_id} ===")
            break

        print(f"\n=== CYCLE {cycle_id} INCOMPLETE ({result.outcome}) — RE-PLANNING ===")
        previous_cycle = result

    return history


def _extract_stage_spec_sections(full_project_goal: str, target_files: list[str]) -> str:
    """
    Extract only the numbered specification sections from `full_project_goal` that
    correspond to `target_files`, preventing Foreman from seeing and scheduling
    future stages' files during an earlier stage.
    """
    if not full_project_goal.strip() or not target_files:
        return full_project_goal.strip()

    norm_targets = {_normalize_workspace_path(tf) for tf in target_files if tf.strip()}
    # Split on numbered top-level sections like `1. config/...`
    blocks = re.split(r"(?=^\s*\d+\.\s+)", full_project_goal.strip(), flags=re.MULTILINE)
    if len(blocks) <= 2:
        return full_project_goal.strip()

    matched_blocks: list[str] = []
    for blk in blocks:
        blk_stripped = blk.strip()
        if not blk_stripped:
            continue
        if any(tf in blk_stripped or Path(tf).name in blk_stripped for tf in norm_targets):
            matched_blocks.append(blk_stripped)

    return "\n\n".join(matched_blocks) if matched_blocks else full_project_goal.strip()


def _format_stage_goal(
    stage: StageGoal,
    total_stages: int,
    full_project_goal: str = "",
) -> str:
    """
    Construct a stage prompt scoped strictly to `stage.target_files` while
    preserving the exact JSON keys, CSV rows, class/method signatures, and formulas
    for those files from `full_project_goal`.
    """
    target_files_text = (
        ", ".join(f"`{f}`" for f in stage.target_files)
        if stage.target_files
        else "as specified below"
    )
    scoped_spec = _extract_stage_spec_sections(full_project_goal, stage.target_files)
    base = (
        f"Build Stage {stage.stage_id} of {total_stages} ({stage.title}).\n"
        f"STRICT STAGE SCOPE: Create/modify ONLY {target_files_text}. Do NOT create or reference files from future stages.\n\n"
        f"Stage {stage.stage_id} Summary:\n{stage.objective.strip()}"
    )
    if scoped_spec and scoped_spec not in stage.objective:
        base += (
            f"\n\nExact Specification for Stage {stage.stage_id} Files ({target_files_text}):\n"
            f"{scoped_spec}"
        )
    return base


def run_staged_project(
    *,
    goal: str,
    architecture: str,
    worker_profile: WorkerProfile,
    stages: list[StageGoal] | None = None,
    max_stages: int = 3,
    max_cycles_per_stage: int = 2,
    resume: bool = False,
    workspace: str | Path | None = None,
    plan_only: bool = False,
    stop_after_stage: int | None = None,
    emit_tests: bool = False,
) -> list[CycleResult]:
    """
    Decompose a complex project into sequential StageGoal milestones (or use provided
    `stages`) and execute each stage inside the target workspace, supporting mid-stage
    checkpointing, `--plan-only`, `--stop-after-stage`, and `--emit-tests`.
    """
    if workspace is not None:
        set_active_workspace(workspace)

    ws_root = get_workspace_root()
    history: list[CycleResult] = []
    completed_stage_ids: list[int] = []
    start_stage_idx = 0
    ckpt = load_checkpoint() if resume else None

    if resume and ckpt is not None and ckpt.stages:
        resolved_stages = list(ckpt.stages)
        start_stage_idx = ckpt.current_stage_index
        completed_stage_ids = list(ckpt.completed_stage_ids)
        history = list(ckpt.completed_cycles)
        print(
            f"\n=== RESUMING MULTI-STAGE PROJECT IN {ws_root} AT STAGE "
            f"{start_stage_idx + 1}/{len(resolved_stages)} "
            f"(Completed Stage IDs: {completed_stage_ids}) ==="
        )
    else:
        clear_checkpoint()
        if stages is not None:
            resolved_stages = list(stages)
        else:
            print(f"\n=== DECOMPOSING PROJECT GOAL INTO SEQUENTIAL STAGES ({ws_root}) ===")
            planner = build_planner()
            blueprint = planner.decompose_stages(
                project_goal=goal,
                architecture_context=architecture,
                max_stages=max_stages,
            )
            print(blueprint.model_dump_json(indent=2))
            resolved_stages = list(blueprint.stages)

    total_stages = len(resolved_stages)
    for stage_idx in range(start_stage_idx, total_stages):
        stage = resolved_stages[stage_idx]
        print(f"\n############################################################")
        print(f"### STAGE {stage.stage_id}/{total_stages}: {stage.title} ({ws_root}) ###")
        print(f"############################################################")

        stage_goal_text = _format_stage_goal(stage, total_stages, goal)
        stage_previous_cycle: CycleResult | None = None
        stage_succeeded = False

        start_cycle = 1
        resuming_mid_stage = (
            resume
            and ckpt is not None
            and stage_idx == start_stage_idx
            and ckpt.phase not in ("initial", "planner_accepted")
        )
        if resuming_mid_stage and ckpt is not None:
            start_cycle = ckpt.cycle_id
            stage_previous_cycle = ckpt.previous_cycle

        for cycle_id in range(start_cycle, max_cycles_per_stage + 1):
            use_resume_cycle = resuming_mid_stage and (cycle_id == start_cycle)
            if not use_resume_cycle:
                placeholder_state: WorkflowState = {
                    "cycle_id": cycle_id,
                    "project_goal": stage_goal_text,
                    "architecture_context": architecture,
                    "worker_profile": worker_profile,
                    "previous_cycle": stage_previous_cycle,
                    "plan": None,
                    "task_batches": [],
                    "task_results": [],
                    "ledger": [],
                    "staged_files": {},
                    "file_snapshots": {},
                    "worker_attempts": {},
                    "foreman_repair_attempts": 0,
                    "rebuild_attempts": 0,
                    "validation_result": None,
                    "foreman_review": None,
                    "planner_review": None,
                    "rework_mode": "plan_only" if plan_only else "none",
                    "phase": "initial",
                }
                save_checkpoint(
                    placeholder_state,
                    completed_cycles=history,
                    stages=resolved_stages,
                    current_stage_index=stage_idx,
                    completed_stage_ids=completed_stage_ids,
                )

            suite_path = f"tests/test_stage_{stage.stage_id}.py"
            cycle_result = run_cycle(
                cycle_id=cycle_id,
                goal=stage_goal_text,
                architecture=architecture,
                worker_profile=worker_profile,
                previous_cycle=stage_previous_cycle,
                resume=use_resume_cycle,
                plan_only=plan_only,
                emit_tests=emit_tests,
                suite_rel_path=suite_path,
            )
            history.append(cycle_result)
            stage_previous_cycle = cycle_result

            if plan_only and cycle_result.outcome == "planned":
                update_checkpoint_stage_metadata(
                    completed_cycles=history,
                    stages=resolved_stages,
                    current_stage_index=stage_idx,
                    completed_stage_ids=completed_stage_ids,
                )
                print(
                    f"\n=== STAGE {stage.stage_id}/{total_stages} PLAN-ONLY COMPLETE ==="
                )
                return history

            if cycle_result.outcome == "overall_success":
                stage_succeeded = True
                if stage.stage_id not in completed_stage_ids:
                    completed_stage_ids.append(stage.stage_id)
                next_idx = min(stage_idx + 1, total_stages - 1)
                update_checkpoint_stage_metadata(
                    completed_cycles=history,
                    stages=resolved_stages,
                    current_stage_index=next_idx,
                    completed_stage_ids=completed_stage_ids,
                )
                print(
                    f"\n=== STAGE {stage.stage_id}/{total_stages} ({stage.title}) ACCEPTED ==="
                )
                break

            update_checkpoint_stage_metadata(
                completed_cycles=history,
                stages=resolved_stages,
                current_stage_index=stage_idx,
                completed_stage_ids=completed_stage_ids,
            )

        if not stage_succeeded:
            print(
                f"\n=== STAGE {stage.stage_id}/{total_stages} FAILED TO CONVERGE — STOPPING ==="
            )
            break

        if stop_after_stage is not None and stage.stage_id >= stop_after_stage:
            print(
                f"\n=== STOPPED AFTER STAGE {stage.stage_id}/{total_stages} (--stop-after-stage {stop_after_stage}) ==="
            )
            break

    return history

from __future__ import annotations

from collections.abc import Callable
import os
import re
import shutil
import subprocess
from pathlib import Path

from .base import CodingHarness
from .config import AiderConfig
from src.orchestration.ledger import (
    ensure_class_wrapper,
    ensure_workspace_imports,
    extract_python_signatures,
    merge_staged_snippets_ast,
    upsert_staged_snippet,
)
from src.orchestration.models import get_worker_model
from src.orchestration.schemas import Plan, Task, TaskBatch, TaskResult
from src.orchestration.workspace import (
    DEFAULT_WORKSPACE_ROOT,
    REPO_ROOT,
    get_access_mode,
    get_external_read_files,
    get_staging_dir,
    get_workspace_root,
    normalize_workspace_rel_path,
    request_external_write_permission,
)

WORKSPACE_ROOT = DEFAULT_WORKSPACE_ROOT
STAGING_DIR = (DEFAULT_WORKSPACE_ROOT / ".staging").resolve()

SUPPORTED_ASSET_EXTENSIONS = {
    ".py",
    ".json",
    ".yaml",
    ".yml",
    ".toml",
    ".ini",
    ".cfg",
    ".csv",
    ".sql",
    ".txt",
    ".md",
}


def _normalize_rel_path(path: str) -> str:
    return normalize_workspace_rel_path(path)


def _module_import_name(rel_path: str) -> str:
    norm = _normalize_rel_path(rel_path)
    p = Path(norm)
    parts = list(p.with_suffix("").parts)
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts) if parts else p.stem


def list_workspace_files() -> list[Path]:
    """
    Return all user code, configuration, and data files in the active workspace,
    excluding hidden dotfiles/directories (such as `.aider.chat.history.md`,
    `.staging`, `.checkpoint`, `.git`, `.pytest_cache`) and `__pycache__`.
    """
    ws_root = get_workspace_root()
    if not ws_root.exists():
        return []
    excluded = {".staging", ".checkpoint", ".git", ".pytest_cache", "__pycache__"}
    results: list[Path] = []
    for p in ws_root.rglob("*"):
        if not p.is_file():
            continue
        rel_parts = p.relative_to(ws_root).parts
        if any(part.startswith(".") or part in excluded for part in rel_parts):
            continue
        if p.suffix.lower() in SUPPORTED_ASSET_EXTENSIONS:
            results.append(p)
    return sorted(results)


def list_workspace_py_files() -> list[Path]:
    """Return all user Python (`.py`) files in the active workspace."""
    return [p for p in list_workspace_files() if p.suffix.lower() == ".py"]


class AiderHarness(CodingHarness):
    """
    Execute a TaskBatch through isolated per-task Aider Worker sessions inside
    the currently active workspace directory.
    """

    def __init__(self, config: AiderConfig | None = None):
        if config is None:
            worker_mod = get_worker_model()
            model_arg = (
                worker_mod
                if worker_mod.startswith(("ollama_chat/", "ollama/"))
                else f"ollama_chat/{worker_mod}"
            )
            self.config = AiderConfig(model=model_arg)
        else:
            self.config = config

    def execute(
        self,
        batch: TaskBatch,
        existing_staged: dict[str, dict[str, str]] | None = None,
        file_snapshots: dict[str, str] | None = None,
        project_goal: str = "",
        verified_task_results: dict[str, TaskResult] | None = None,
        on_task_complete: Callable[[Task, TaskResult, dict[str, dict[str, str]]], None] | None = None,
        stop_after_tasks: int | None = None,
        plan: Plan | None = None,
    ) -> list[TaskResult]:
        if not batch.tasks:
            raise ValueError("TaskBatch must contain at least one task.")

        ws_root = get_workspace_root()
        request_external_write_permission(ws_root, operation='execute Aider worker tasks in workspace')
        staging_dir = get_staging_dir()
        ws_root.mkdir(parents=True, exist_ok=True)
        staging_dir.mkdir(parents=True, exist_ok=True)

        sandbox_git = ws_root / ".git"
        if not sandbox_git.exists():
            subprocess.run(
                ["git", "init", "-q"],
                cwd=ws_root,
                check=False,
            )

        tasks_by_id = {task.task_id: task for task in batch.tasks}
        execution_ids = batch.execution_order or [t.task_id for t in batch.tasks]

        ordered_tasks: list[Task] = []
        for task_id in execution_ids:
            task = tasks_by_id.get(task_id)
            if task is not None:
                ordered_tasks.append(task)

        seen_ids = {t.task_id for t in ordered_tasks}
        for task in batch.tasks:
            if task.task_id not in seen_ids:
                ordered_tasks.append(task)

        file_task_counts: dict[str, int] = {}
        for task in ordered_tasks:
            for raw_path in (task.relevant_files or batch.relevant_files):
                norm = _normalize_rel_path(raw_path)
                file_task_counts[norm] = file_task_counts.get(norm, 0) + 1

        staged_snippets: dict[str, dict[str, str]] = {
            _normalize_rel_path(k): dict(v) for k, v in (existing_staged or {}).items()
        }
        snapshots = {_normalize_rel_path(k): v for k, v in (file_snapshots or {}).items()}
        verified_map = dict(verified_task_results or {})

        env = os.environ.copy()
        env.pop("GIT_DIR", None)
        env.pop("GIT_WORK_TREE", None)
        env["AIDER_ANALYTICS"] = "false"
        env["AIDER_CHECK_UPDATE"] = "false"
        env["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
        env["POSTHOG_DISABLED"] = "1"
        env["TELEMETRY_DISABLED"] = "1" 

        env_stop = os.environ.get("ORCHESTRATOR_STOP_AFTER_TASKS")
        if stop_after_tasks is None and env_stop and env_stop.isdigit():
            stop_after_tasks = int(env_stop)

        task_results: list[TaskResult] = []
        completed_files: set[str] = set()
        completed_sibling_sigs: dict[str, list[str]] = {}
        newly_executed_count = 0

        for mod_file, snippets_map in staged_snippets.items():
            for prev_tid, prev_code in snippets_map.items():
                for sig in extract_python_signatures(prev_code):
                    entry_str = f"{sig} (from task {prev_tid})"
                    if entry_str not in completed_sibling_sigs.setdefault(mod_file, []):
                        completed_sibling_sigs[mod_file].append(entry_str)

        try:
            for task in ordered_tasks:
                declared_files = sorted(
                    {
                        _normalize_rel_path(p)
                        for p in (task.relevant_files or batch.relevant_files)
                        if _normalize_rel_path(p)
                    }
                )
                for path in declared_files:
                    abs_target = self._safe_workspace_path(path)
                    abs_target.parent.mkdir(parents=True, exist_ok=True)
                    if path.endswith(".py"):
                        self._ensure_package_inits(abs_target.parent)

                primary_target = declared_files[0]
                is_python_target = primary_target.endswith(".py")
                use_staging = is_python_target and (
                    file_task_counts.get(primary_target, 0) > 1
                    or primary_target in staged_snippets
                )

                prev_res = verified_map.get(task.task_id)
                if prev_res is not None and prev_res.status == "completed":
                    print(
                        f"\n=== AIDER WORKER (Task: {task.task_id}) — SKIPPING (Verified on disk from checkpoint) ==="
                    )
                    for f in prev_res.files_changed:
                        completed_files.add(_normalize_rel_path(f))
                    completed_files.add(primary_target)
                    task_results.append(prev_res)
                    continue

                if use_staging:
                    safe_name = primary_target.replace("/", "__")
                    staged_rel_path = f".staging/{task.task_id}__{safe_name}"
                    staged_abs_path = self._safe_workspace_path(staged_rel_path)
                    staged_abs_path.parent.mkdir(parents=True, exist_ok=True)
                    staged_abs_path.write_text("", encoding="utf-8")
                    editable_files = [staged_rel_path]
                else:
                    staged_rel_path = None
                    editable_files = list(declared_files)
                    if len(editable_files) > 1:
                        uncompleted = [
                            f for f in editable_files if f not in completed_files
                        ]
                        if uncompleted:
                            editable_files = uncompleted

                read_only_files = [
                    f
                    for f in _resolve_task_read_only_files(
                        task=task,
                        batch=batch,
                        primary_target=primary_target,
                        ws_root=ws_root,
                        plan=plan,
                    )
                    if f not in editable_files
                    and (not use_staging or f != primary_target)
                    and self._is_non_empty_file(f)
                ]

                before = self._snapshot_files(editable_files)
                before_primary = (
                    self._snapshot_files([primary_target]).get(primary_target, "")
                    if use_staging
                    else ""
                )

                prompt = self._build_task_prompt(
                    batch=batch,
                    task=task,
                    logical_target=primary_target,
                    editable_files=editable_files,
                    read_only_files=read_only_files,
                    use_staging=use_staging,
                    sibling_signatures=completed_sibling_sigs.get(primary_target, []),
                    project_goal=project_goal,
                    plan=plan,
                )

                ext_read_paths = (
                    [str(p) for p in get_external_read_files() if p.is_file()]
                    if get_access_mode() in ("read_external", "allow_external")
                    else []
                )
                read_args: list[str] = []
                for rf in [*read_only_files, *ext_read_paths]:
                    read_args.extend(["--read", rf])

                command = [
                    "aider",
                    "--model",
                    self.config.model,
                    "--edit-format",
                    self.config.edit_format,
                    "--yes",
                    "--no-auto-commits",
                    "--no-auto-lint",
                    "--no-git",
                    "--no-analytics",
                    "--no-check-update",
                    "--no-show-model-warnings",
                    *read_args,
                    "--message",
                    prompt,
                    *editable_files,
                ]

                print(f"\n=== AIDER WORKER (Task: {task.task_id}) ===")
                print(f"Workspace: {ws_root}")
                print(f"Model: {self.config.model}")
                print(f"Batch: {batch.batch_id}")
                print(f"Target File: {primary_target}")
                if task.target_symbol:
                    print(f"Target Symbol: {task.target_symbol}")
                print(f"Editable File: {', '.join(editable_files)}")
                if read_only_files:
                    print(f"Read-Only Context: {', '.join(read_only_files)}")

                completed = subprocess.run(
                    command,
                    cwd=ws_root,
                    env=env,
                    text=True,
                )

                after = self._snapshot_files(editable_files)
                changed_physical = [
                    f for f in editable_files if before.get(f) != after.get(f)
                ]

                if use_staging and staged_rel_path:
                    raw_snippet = after.get(staged_rel_path, "").strip()
                    if not raw_snippet:
                        staged_fname = Path(staged_rel_path).name
                        for stray_cand in (
                            ws_root / f".{staged_fname}",
                            ws_root / staged_fname,
                            staging_dir / f".{staged_fname}",
                            ws_root / f".path/to/{staged_rel_path}",
                            ws_root / f".path/to/{staged_fname}",
                            ws_root / f"path/to/{staged_rel_path}",
                            ws_root / f"path/to/{ primary_target }",
                        ):
                            if stray_cand.is_file() and stray_cand.read_text(encoding="utf-8").strip():
                                raw_snippet = stray_cand.read_text(encoding="utf-8").strip()
                                stray_cand.unlink(missing_ok=True)
                                self._safe_workspace_path(staged_rel_path).write_text(
                                    raw_snippet + "\n", encoding="utf-8"
                                )
                                break
                    if not raw_snippet:
                        for cand_sub in (
                            staging_dir / primary_target,
                            staging_dir / "path/to" / primary_target,
                            staging_dir / ".path/to" / primary_target,
                            staging_dir / "path/to" / Path(staged_rel_path).name,
                            staging_dir / ".path/to" / Path(staged_rel_path).name,
                        ):
                            if cand_sub.is_file() and cand_sub.read_text(encoding="utf-8").strip():
                                raw_snippet = cand_sub.read_text(encoding="utf-8").strip()
                                cand_sub.unlink(missing_ok=True)
                                self._safe_workspace_path(staged_rel_path).write_text(
                                    raw_snippet + "\n", encoding="utf-8"
                                )
                                break
                        for stray_dir in (staging_dir / "path", staging_dir / ".path", ws_root / "path", ws_root / ".path"):
                            if stray_dir.is_dir():
                                shutil.rmtree(stray_dir, ignore_errors=True)
                    if not raw_snippet:
                        after_primary = (
                            self._snapshot_files([primary_target])
                            .get(primary_target, "")
                            .strip()
                        )
                        if after_primary and after_primary != before_primary.strip():
                            raw_snippet = after_primary
                            self._safe_workspace_path(staged_rel_path).write_text(
                                raw_snippet + "\n", encoding="utf-8"
                            )

                    snippet_code = ensure_class_wrapper(raw_snippet, task.target_symbol).strip()
                    if snippet_code and snippet_code != raw_snippet:
                        self._safe_workspace_path(staged_rel_path).write_text(
                            snippet_code + "\n", encoding="utf-8"
                        )
                    changed = [primary_target] if snippet_code else []
                    if snippet_code:
                        staged_snippets[primary_target] = upsert_staged_snippet(
                            staged_snippets.get(primary_target, {}),
                            task.task_id,
                            snippet_code,
                            target_symbol=task.target_symbol,
                        )
                        sigs = extract_python_signatures(snippet_code)
                        if sigs:
                            for s in sigs:
                                entry_str = f"{s} (from task {task.task_id})"
                                if entry_str not in completed_sibling_sigs.setdefault(primary_target, []):
                                    completed_sibling_sigs[primary_target].append(entry_str)
                        ordered_pairs = list(staged_snippets[primary_target].items())
                        merged_live = merge_staged_snippets_ast(
                            target_filename=primary_target,
                            base_source=snapshots.get(primary_target, ""),
                            ordered_snippets=ordered_pairs,
                        )
                        target_abs = self._safe_workspace_path(primary_target)
                        target_abs.parent.mkdir(parents=True, exist_ok=True)
                        self._ensure_package_inits(target_abs.parent)
                        target_abs.write_text(merged_live, encoding="utf-8")
                        completed_files.add(primary_target)
                else:
                    changed = changed_physical
                    for f in changed:
                        if f.endswith(".py"):
                            abs_f = self._safe_workspace_path(f)
                            if abs_f.is_file():
                                fixed_src = ensure_workspace_imports(
                                    abs_f.read_text(encoding="utf-8"), f, ws_root
                                )
                                abs_f.write_text(fixed_src, encoding="utf-8")
                        completed_files.add(f)

                errors: list[str] = []
                if completed.returncode != 0:
                    errors.append(
                        f"Aider exited with non-zero status: {completed.returncode}"
                    )
                    status = "failure"
                elif changed:
                    status = "completed"
                elif not use_staging and self._is_non_empty_file(primary_target):
                    changed = [primary_target]
                    completed_files.add(primary_target)
                    status = "completed"
                else:
                    status = "no_change"

                result_obj = TaskResult(
                    task_id=task.task_id,
                    status=status,
                    summary=(
                        f"Executed task {task.task_id} "
                        f"({'staged for ' + primary_target if use_staging else primary_target})."
                    ),
                    files_changed=changed,
                    staged_file=staged_rel_path if use_staging else None,
                    errors=errors,
                )
                task_results.append(result_obj)
                newly_executed_count += 1

                if on_task_complete is not None:
                    on_task_complete(task, result_obj, staged_snippets)

                if stop_after_tasks is not None and newly_executed_count >= stop_after_tasks:
                    raise KeyboardInterrupt(
                        f"Stopped after {newly_executed_count} task(s) completed (last={task.task_id})."
                    )
        finally:
            self._unload_worker_model()

        return task_results

    def _ensure_package_inits(self, directory: Path) -> None:
        """Ensure every intermediate Python package directory inside the active workspace has an __init__.py."""
        ws_root = get_workspace_root()
        curr = directory.resolve()
        while curr != ws_root and curr.is_relative_to(ws_root):
            if curr.name not in (".staging", ".checkpoint", ".git", "__pycache__", ".pytest_cache"):
                init_file = curr / "__init__.py"
                if not init_file.exists():
                    init_file.write_text("", encoding="utf-8")
            curr = curr.parent

    def _is_non_empty_file(self, path: str) -> bool:
        p = self._safe_workspace_path(path)
        return p.is_file() and bool(p.read_text(encoding="utf-8").strip())

    def _unload_worker_model(self) -> None:
        raw_model = self.config.model
        for prefix in ("ollama_chat/", "ollama/"):
            if raw_model.startswith(prefix):
                raw_model = raw_model.removeprefix(prefix)

        subprocess.run(
            ["ollama", "stop", raw_model],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )

    def _safe_workspace_path(self, path: str) -> Path:
        ws_root = get_workspace_root()
        normalized = _normalize_rel_path(path)
        requested = (ws_root / normalized).resolve()
        if not requested.is_relative_to(ws_root):
            raise ValueError(f"Path {path} escapes workspace root {ws_root}")
        return requested

    def _build_task_prompt(
        self,
        *,
        batch: TaskBatch,
        task: Task,
        logical_target: str,
        editable_files: list[str],
        read_only_files: list[str],
        use_staging: bool,
        sibling_signatures: list[str],
        project_goal: str = "",
        plan: Plan | None = None,
    ) -> str:
        is_python = logical_target.endswith(".py")
        target_import_name = _module_import_name(logical_target) if is_python else ""
        lines = [
            f"# Batch Objective: {batch.objective}",
        ]
        scoped_file_spec = _extract_file_spec_section(project_goal, logical_target)
        if scoped_file_spec:
            lines.extend(
                [
                    f"## Specification for `{logical_target}`:",
                    scoped_file_spec,
                    "",
                ]
            )
        lines.append(f"## Current Atomic Task {task.task_id}: {task.objective}")
        if is_python:
            lines.append(f"Logical Module: {logical_target} (import path: `{target_import_name}`)")
        else:
            lines.append(
                f"Target Asset File: {logical_target} (format: `{Path(logical_target).suffix.lstrip('.').upper()}`)"
            )
        if task.target_symbol:
            lines.append(f"Target Function/Symbol to Implement: {task.target_symbol}")
        lines.extend(
            [
                f"File to Edit Now: {', '.join(editable_files)}",
                f"Scope: {task.scope}",
                "",
            ]
        )

        if not is_python:
            lines.extend(
                [
                    "## Non-Python Config / Data File Mode:",
                    f"  - Write ONLY valid `{Path(logical_target).suffix}` content into `{editable_files[0]}`.",
                    "  - Do NOT wrap the file content in Python code or extra commentary.",
                    "",
                ]
            )
        elif use_staging:
            if task.target_symbol and "." in task.target_symbol:
                cls_name, method_name = task.target_symbol.split(".", 1)
                lines.extend(
                    [
                        f"## Class Method Staging Mode (`{cls_name}.{method_name}`):",
                        f"  - Write `class {cls_name}:` containing ONLY the `{method_name}` method into `{editable_files[0]}`.",
                        f"  - Do NOT write `pass` stubs for other methods of `{cls_name}` — the Foreman will automatically merge all staged methods into `class {cls_name}:` in `{logical_target}`.",
                        f"  - Do NOT write `from {target_import_name} import ...` or `from {Path(logical_target).stem} import ...` because your code will be placed directly inside `{logical_target}`.",
                        "  - If your method uses classes/functions from ANOTHER module (listed in Read-Only Reference Files), include the required `from <package.module> import ...` at the top of your snippet.",
                        "",
                    ]
                )
            else:
                lines.extend(
                    [
                        "## Staging & Assembly Mode:",
                        f"  - Write ONLY the code required for Task {task.task_id} "
                        f"({task.target_symbol or task.objective}) into `{editable_files[0]}`.",
                        f"  - The Foreman will assemble `{editable_files[0]}` into `{logical_target}`.",
                        f"  - Do NOT write `from {target_import_name} import ...` or `from {Path(logical_target).stem} import ...` because your code will be placed directly inside `{logical_target}`.",
                        "  - If your function calls functions from ANOTHER module (listed in Read-Only Reference Files), include the required `from <package.module> import ...` at the top of your snippet.",
                        "",
                    ]
                )

        if plan is not None and getattr(plan, "file_blueprints", None):
            for fb in plan.file_blueprints:
                if _normalize_rel_path(fb.path) == logical_target:
                    lines.append(f"## Planner Architectural Blueprint for `{logical_target}`:")
                    lines.append(f"  - Role: {fb.role}")
                    if fb.required_imports:
                        lines.append(f"  - Required Imports: {', '.join(fb.required_imports)}")
                    if fb.symbols:
                        lines.append("  - Planned Module Symbols:")
                        for sym in fb.symbols:
                            marker = " (CURRENT TASK)" if sym.target_symbol == task.target_symbol else ""
                            lines.append(f"    * `{sym.target_symbol}`{marker}: `{sym.signature}` -> {sym.contract}")
                    lines.append("")
                    break

        if sibling_signatures:
            lines.append(
                f"## Sibling Functions/Methods Already Implemented in `{logical_target}` (you may reference these attributes & methods directly):"
            )
            for sig in sibling_signatures:
                lines.append(f"  - {sig}")
            lines.append("")

        if read_only_files:
            ro_descriptions = [
                f"{rf} (import as `{_module_import_name(rf)}`)"
                if rf.endswith(".py")
                else f"{rf} (data/config file)"
                for rf in read_only_files
            ]
            lines.append(
                f"Read-Only Reference Files (do NOT edit): {', '.join(ro_descriptions)}"
            )
            lines.append("")

        if task.constraints or batch.constraints:
            lines.append("## Constraints:")
            for c in batch.constraints + task.constraints:
                lines.append(f"  - {c}")
            lines.append("")

        if task.acceptance_criteria:
            lines.append("## Acceptance Criteria:")
            for ac in task.acceptance_criteria:
                lines.append(f"  - {ac}")
            lines.append("")

        if task.test_assertions:
            lines.append(
                "## Verification Assertions (run automatically by Validator — do NOT paste these into the module unless requested):"
            )
            for assertion in task.test_assertions:
                lines.append(f"  - {assertion}")
            lines.append("")

        lines.append(
            f"Modify ONLY `{editable_files[0]}`. "
            "Make minimal, focused changes that satisfy the acceptance criteria."
        )
        lines.append("Do not add extra code, test blocks, or wrappers beyond what is specified.")
        return "\n".join(lines)

    def _snapshot_files(self, files: list[str]) -> dict[str, str]:
        snapshot = {}
        for f in files:
            path = self._safe_workspace_path(f)
            if path.exists() and path.is_file():
                snapshot[f] = path.read_text(encoding="utf-8")
            else:
                snapshot[f] = ""
        return snapshot


def _resolve_task_read_only_files(
    task: Task,
    batch: TaskBatch,
    primary_target: str,
    ws_root: Path,
    plan: Plan | None,
) -> list[str]:
    """Resolve direct imported Python dependencies and required config assets, excluding tests/."""
    referenced: set[str] = set()

    if plan and getattr(plan, "file_blueprints", None):
        for fb in plan.file_blueprints:
            if _normalize_rel_path(fb.path) == primary_target:
                for imp in fb.required_imports:
                    cleaned = imp.strip()
                    if cleaned.startswith("from "):
                        mod_token = cleaned.removeprefix("from ").split()[0].strip()
                    elif cleaned.startswith("import "):
                        mod_token = cleaned.removeprefix("import ").split()[0].split(",")[0].strip()
                    else:
                        continue
                    cand_py = mod_token.replace(".", "/") + ".py"
                    if (ws_root / cand_py).is_file():
                        referenced.add(_normalize_rel_path(cand_py))

    tasks_by_id = {t.task_id: t for t in batch.tasks}
    for dep_id in task.dependencies:
        dep_task = tasks_by_id.get(dep_id)
        if dep_task:
            for f in dep_task.relevant_files:
                norm_f = _normalize_rel_path(f)
                if norm_f != primary_target and (ws_root / norm_f).is_file():
                    referenced.add(norm_f)

    # Also scan task objective, acceptance criteria, and assertions for references to existing workspace .py modules
    task_text = f"{task.objective} {' '.join(task.acceptance_criteria)} {' '.join(task.test_assertions)}"
    for py_path in list_workspace_py_files():
        rel_py = str(py_path.relative_to(ws_root))
        if rel_py == primary_target or rel_py.startswith("tests/") or py_path.name in ("__init__.py", "main.py", "app.py"):
            continue
        if py_path.stem in task_text or rel_py in task_text:
            referenced.add(rel_py)
        else:
            try:
                src = py_path.read_text(encoding="utf-8")
                for sig in extract_python_signatures(src):
                    sym_name = sig.split("(")[0].split("[")[0].removeprefix("def ").removeprefix("class ").strip()
                    if sym_name and sym_name in task_text:
                        referenced.add(rel_py)
                        break
            except Exception:
                pass

    # For any Python task, include existing non-Python config/data assets (.json, .csv, .toml, .ini)
    # under 4KB so the Worker always sees exact config keys and CSV column headers.
    if primary_target.endswith(".py"):
        for asset_path in list_workspace_files():
            rel_asset = str(asset_path.relative_to(ws_root))
            if rel_asset == primary_target or rel_asset.startswith("tests/"):
                continue
            if asset_path.suffix.lower() in (".json", ".csv", ".toml", ".yaml", ".yml", ".ini", ".cfg"):
                try:
                    if asset_path.stat().st_size <= 4096:
                        referenced.add(rel_asset)
                except Exception:
                    pass

    return sorted(
        f for f in referenced
        if f != primary_target
        and not f.startswith("tests/")
        and not f.endswith("__init__.py")
    )


def _extract_file_spec_section(full_project_goal: str, target_file: str) -> str:
    """Extract only the specification block whose header matches `target_file`."""
    if not full_project_goal.strip():
        return ""
    fname = Path(target_file).name
    cleaned_goal = full_project_goal.replace("\u200b", "").replace("\ufeff", "").strip()
    blocks = re.split(r"(?=^\s*\d+\.\s+)", cleaned_goal, flags=re.MULTILINE)
    if len(blocks) <= 2:
        return full_project_goal.strip()
    header_matched = []
    for blk in blocks:
        blk_s = blk.strip()
        if not blk_s or not re.match(r"^\d+\.\s+", blk_s):
            continue
        header = blk_s.splitlines()[0].split(":", 1)[0]
        if target_file in header or fname in header:
            header_matched.append(blk_s)
    if header_matched:
        return "\n\n".join(header_matched)
    matched = [
        blk.strip()
        for blk in blocks
        if blk.strip() and (target_file in blk or fname in blk)
    ]
    return "\n\n".join(matched) if matched else full_project_goal.strip()

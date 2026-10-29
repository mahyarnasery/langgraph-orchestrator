from __future__ import annotations

from collections.abc import Callable
import os
import subprocess
from pathlib import Path

from .base import CodingHarness
from .config import AiderConfig
from src.orchestration.ledger import (
    extract_python_signatures,
    merge_staged_snippets_ast,
    upsert_staged_snippet,
)
from src.orchestration.schemas import Task, TaskBatch, TaskResult


REPO_ROOT = Path("/mnt/ai/langgraph-orchestrator").resolve()
WORKSPACE_ROOT = (REPO_ROOT / "sandbox").resolve()
STAGING_DIR = (WORKSPACE_ROOT / ".staging").resolve()


def _normalize_rel_path(path: str) -> str:
    cleaned = path.strip().replace("\\", "/")
    while cleaned.startswith("./"):
        cleaned = cleaned[2:]
    cleaned = cleaned.removeprefix("sandbox/").strip("/")
    return cleaned


def _module_import_name(rel_path: str) -> str:
    norm = _normalize_rel_path(rel_path)
    p = Path(norm)
    parts = list(p.with_suffix("").parts)
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts) if parts else p.stem


def list_workspace_py_files() -> list[Path]:
    """Return all user Python files in WORKSPACE_ROOT, excluding .staging, .checkpoint, .git, and .pytest_cache."""
    if not WORKSPACE_ROOT.exists():
        return []
    excluded = {".staging", ".checkpoint", ".git", ".pytest_cache", "__pycache__"}
    return sorted(
        p
        for p in WORKSPACE_ROOT.rglob("*.py")
        if not (set(p.parts) & excluded)
    )


class AiderHarness(CodingHarness):
    """
    Execute a TaskBatch through isolated per-task Aider Worker sessions.

    - Supports intra-batch checkpointing (`on_task_complete`) and skipping tasks
      already verified on disk during a resumed session (`verified_task_results`).
    - If multiple tasks in a batch target the same file (function-level atomization),
      each task is executed in an isolated staging file (`sandbox/.staging/<task_id>__<file>`)
      and live-merged into `sandbox/<file>` so the 3B Worker only writes one function at a time.
    """

    def __init__(self, config: AiderConfig | None = None):
        self.config = config or AiderConfig()

    def execute(
        self,
        batch: TaskBatch,
        existing_staged: dict[str, dict[str, str]] | None = None,
        file_snapshots: dict[str, str] | None = None,
        project_goal: str = "",
        verified_task_results: dict[str, TaskResult] | None = None,
        on_task_complete: Callable[[Task, TaskResult, dict[str, dict[str, str]]], None] | None = None,
        stop_after_tasks: int | None = None,
    ) -> list[TaskResult]:
        if not batch.tasks:
            raise ValueError("TaskBatch must contain at least one task.")

        WORKSPACE_ROOT.mkdir(parents=True, exist_ok=True)
        STAGING_DIR.mkdir(parents=True, exist_ok=True)

        sandbox_git = WORKSPACE_ROOT / ".git"
        if not sandbox_git.exists():
            subprocess.run(
                ["git", "init", "-q"],
                cwd=WORKSPACE_ROOT,
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

        env_stop = os.environ.get("ORCHESTRATOR_STOP_AFTER_TASKS")
        if stop_after_tasks is None and env_stop and env_stop.isdigit():
            stop_after_tasks = int(env_stop)

        task_results: list[TaskResult] = []
        completed_files: set[str] = set()
        completed_sibling_sigs: dict[str, list[str]] = {}
        newly_executed_count = 0

        # Pre-populate sibling signatures from already-staged snippets (for rebuild/resumed batches)
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
                    self._ensure_package_inits(abs_target.parent)

                primary_target = declared_files[0]
                use_staging = (
                    file_task_counts.get(primary_target, 0) > 1
                    or primary_target in staged_snippets
                )

                # Check if this task was already completed and verified on disk (Resume Fast-Forward)
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

                dep_files: set[str] = set()
                for dep_id in task.dependencies:
                    dep_task = tasks_by_id.get(dep_id)
                    if dep_task:
                        for f in dep_task.relevant_files:
                            dep_files.add(_normalize_rel_path(f))

                existing_workspace_py = {
                    str(p.relative_to(WORKSPACE_ROOT))
                    for p in list_workspace_py_files()
                }
                read_only_files = sorted(
                    f
                    for f in (
                        dep_files
                        | completed_files
                        | set(declared_files)
                        | existing_workspace_py
                    )
                    if f not in editable_files
                    and (not use_staging or f != primary_target)
                    and self._is_non_empty_file(f)
                )

                before = self._snapshot_files(editable_files)

                prompt = self._build_task_prompt(
                    batch=batch,
                    task=task,
                    logical_target=primary_target,
                    editable_files=editable_files,
                    read_only_files=read_only_files,
                    use_staging=use_staging,
                    sibling_signatures=completed_sibling_sigs.get(primary_target, []),
                    project_goal=project_goal,
                )

                read_args: list[str] = []
                for rf in read_only_files:
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
                    "--subtree-only",
                    *read_args,
                    "--message",
                    prompt,
                    *editable_files,
                ]

                print(f"\n=== AIDER WORKER (Task: {task.task_id}) ===")
                print(f"Model: {self.config.model}")
                print(f"Batch: {batch.batch_id}")
                print(f"Target Module: {primary_target}")
                if task.target_symbol:
                    print(f"Target Symbol: {task.target_symbol}")
                print(f"Editable File: {', '.join(editable_files)}")
                if read_only_files:
                    print(f"Read-Only Context: {', '.join(read_only_files)}")

                completed = subprocess.run(
                    command,
                    cwd=WORKSPACE_ROOT,
                    env=env,
                    text=True,
                )

                after = self._snapshot_files(editable_files)
                changed_physical = [
                    f for f in editable_files if before.get(f) != after.get(f)
                ]

                if use_staging and staged_rel_path:
                    snippet_code = after.get(staged_rel_path, "").strip()
                    changed = [primary_target] if snippet_code else []
                    if snippet_code:
                        staged_snippets[primary_target] = upsert_staged_snippet(
                            staged_snippets.get(primary_target, {}),
                            task.task_id,
                            snippet_code,
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
                        completed_files.add(f)

                errors: list[str] = []
                if completed.returncode != 0:
                    errors.append(
                        f"Aider exited with non-zero status: {completed.returncode}"
                    )
                    status = "failure"
                elif changed:
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
                        f"Simulated interrupt after {newly_executed_count} task(s) completed (last={task.task_id})."
                    )
        finally:
            self._unload_worker_model()

        return task_results

    def _ensure_package_inits(self, directory: Path) -> None:
        """Ensure every intermediate package directory inside WORKSPACE_ROOT has an __init__.py."""
        curr = directory.resolve()
        while curr != WORKSPACE_ROOT and curr.is_relative_to(WORKSPACE_ROOT):
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
        normalized = _normalize_rel_path(path)
        requested = (WORKSPACE_ROOT / normalized).resolve()
        if not requested.is_relative_to(WORKSPACE_ROOT):
            raise ValueError(f"Path {path} escapes workspace root")
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
    ) -> str:
        target_import_name = _module_import_name(logical_target)
        lines = [
            f"# Batch Objective: {batch.objective}",
        ]
        if project_goal.strip():
            lines.extend(
                [
                    "## Full Project Specification (for exact formula & signature reference):",
                    project_goal.strip(),
                    "",
                ]
            )
        lines.extend(
            [
                f"## Current Atomic Task {task.task_id}: {task.objective}",
                f"Logical Module: {logical_target} (import path: `{target_import_name}`)",
            ]
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

        if use_staging:
            lines.extend(
                [
                    "## Staging & Assembly Mode:",
                    f"  - Write ONLY the code required for Task {task.task_id} "
                    f"({task.target_symbol or task.objective}) into {', '.join(editable_files)}.",
                    f"  - The Foreman will assemble {', '.join(editable_files)} into `{logical_target}`.",
                    f"  - Do NOT write `from {target_import_name} import ...` or `from {Path(logical_target).stem} import ...` because your code will be placed directly inside `{logical_target}`.",
                    "  - If your function calls functions from ANOTHER module (listed in Read-Only Reference Files), include the required `from <package.module> import ...` at the top of your snippet.",
                    "",
                ]
            )

        if sibling_signatures:
            lines.append(
                f"## Sibling Functions Already Implemented in `{logical_target}` (you may call these directly):"
            )
            for sig in sibling_signatures:
                lines.append(f"  - {sig}")
            lines.append("")

        if read_only_files:
            ro_descriptions = [
                f"{rf} (import as `{_module_import_name(rf)}`)" for rf in read_only_files
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
            f"Modify ONLY {', '.join(editable_files)}. "
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

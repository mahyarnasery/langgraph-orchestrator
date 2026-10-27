from __future__ import annotations

import os
import subprocess
from pathlib import Path

from .base import CodingHarness
from .config import AiderConfig
from src.orchestration.schemas import Task, TaskBatch, TaskResult


REPO_ROOT = Path("/mnt/ai/langgraph-orchestrator").resolve()
WORKSPACE_ROOT = (REPO_ROOT / "sandbox").resolve()


class AiderHarness(CodingHarness):
    """
    Execute a TaskBatch through isolated per-task Aider Worker sessions.

    The Worker is strictly confined to WORKSPACE_ROOT.
    Upstream dependency files are passed via `--read` so the Worker has
    exact signature awareness without risking accidental edits to completed files.
    """

    def __init__(self, config: AiderConfig | None = None):
        self.config = config or AiderConfig()

    def execute(self, batch: TaskBatch) -> list[TaskResult]:
        if not batch.tasks:
            raise ValueError("TaskBatch must contain at least one task.")

        if not WORKSPACE_ROOT.is_dir():
            WORKSPACE_ROOT.mkdir(parents=True, exist_ok=True)

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

        all_files = sorted(
            {
                path.removeprefix("sandbox/")
                for task in ordered_tasks
                for path in (task.relevant_files or batch.relevant_files)
            }
            | {path.removeprefix("sandbox/") for path in batch.relevant_files}
        )

        if not all_files:
            raise ValueError("TaskBatch must identify at least one relevant file.")

        for path in all_files:
            self._safe_workspace_path(path)

        env = os.environ.copy()
        env.pop("GIT_DIR", None)
        env.pop("GIT_WORK_TREE", None)

        task_results: list[TaskResult] = []
        completed_files: set[str] = set()

        try:
            for task in ordered_tasks:
                declared_files = sorted(
                    {
                        p.removeprefix("sandbox/")
                        for p in (task.relevant_files or batch.relevant_files)
                    }
                )

                # Separate editable target files from already-completed dependency files
                editable_files = list(declared_files)
                if len(editable_files) > 1:
                    uncompleted = [
                        f for f in editable_files if f not in completed_files
                    ]
                    if uncompleted:
                        editable_files = uncompleted

                # Gather read-only context files from dependencies & previously completed tasks
                dep_files: set[str] = set()
                for dep_id in task.dependencies:
                    dep_task = tasks_by_id.get(dep_id)
                    if dep_task:
                        for f in dep_task.relevant_files:
                            dep_files.add(f.removeprefix("sandbox/"))

                existing_workspace_py = {
                    str(p.relative_to(WORKSPACE_ROOT))
                    for p in WORKSPACE_ROOT.rglob("*.py")
                }
                read_only_files = sorted(
                    f
                    for f in (
                        dep_files
                        | completed_files
                        | set(declared_files)
                        | existing_workspace_py
                    )
                    if f not in editable_files and self._is_non_empty_file(f)
                )

                before = self._snapshot_files(editable_files)

                prompt = self._build_task_prompt(
                    batch=batch,
                    task=task,
                    editable_files=editable_files,
                    read_only_files=read_only_files,
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
                print(f"Editable Files: {', '.join(editable_files)}")
                if read_only_files:
                    print(f"Read-Only Context: {', '.join(read_only_files)}")

                completed = subprocess.run(
                    command,
                    cwd=WORKSPACE_ROOT,
                    env=env,
                    text=True,
                )

                after = self._snapshot_files(editable_files)
                changed = [
                    f for f in editable_files if before.get(f) != after.get(f)
                ]

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

                task_results.append(
                    TaskResult(
                        task_id=task.task_id,
                        status=status,
                        summary=(
                            f"Executed task {task.task_id} via Aider "
                            f"({len(changed)} file(s) modified)."
                        ),
                        files_changed=changed,
                        errors=errors,
                    )
                )
        finally:
            self._unload_worker_model()

        return task_results

    def _is_non_empty_file(self, path: str) -> bool:
        p = self._safe_workspace_path(path)
        return p.is_file() and bool(p.read_text(encoding="utf-8").strip())

    def _unload_worker_model(self) -> None:
        """Stop the Ollama worker model after execution to free VRAM for Foreman/Planner."""
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
        """Resolve a path relative to the workspace root, preventing directory traversal."""
        normalized = path.removeprefix("sandbox/")
        requested = (WORKSPACE_ROOT / normalized).resolve()
        if not requested.is_relative_to(WORKSPACE_ROOT):
            raise ValueError(f"Path {path} escapes workspace root")
        return requested

    def _build_task_prompt(
        self,
        *,
        batch: TaskBatch,
        task: Task,
        editable_files: list[str],
        read_only_files: list[str],
    ) -> str:
        """Build a focused single-task prompt for Aider."""
        lines = [
            f"# Batch Objective: {batch.objective}",
            f"## Task {task.task_id}: {task.objective}",
            f"Target File(s) to Edit: {', '.join(editable_files)}",
            f"Scope: {task.scope}",
            "",
        ]

        if read_only_files:
            lines.append(
                f"Read-Only Reference Files (do NOT edit): {', '.join(read_only_files)}"
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

        if batch.shared_context:
            lines.append("## Shared Context:")
            for ctx in batch.shared_context:
                lines.append(f"  - {ctx}")
            lines.append("")

        lines.append(
            f"Modify ONLY {', '.join(editable_files)}. "
            "Make minimal, focused changes that satisfy the acceptance criteria."
        )
        lines.append("Do not add extra code, comments, or wrappers beyond what is specified.")
        return "\n".join(lines)

    def _snapshot_files(self, files: list[str]) -> dict[str, str]:
        """Capture current content of each file (empty string if not exists)."""
        snapshot = {}
        for f in files:
            path = self._safe_workspace_path(f)
            if path.exists() and path.is_file():
                snapshot[f] = path.read_text(encoding="utf-8")
            else:
                snapshot[f] = ""
        return snapshot

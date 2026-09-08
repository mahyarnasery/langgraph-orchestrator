from __future__ import annotations

import hashlib
import os
import subprocess
from pathlib import Path

from .base import CodingHarness
from .config import AiderConfig
from src.orchestration.schemas import TaskBatch, TaskResult


REPO_ROOT = Path("/mnt/ai/langgraph-orchestrator").resolve()
WORKSPACE_ROOT = (REPO_ROOT / "sandbox").resolve()


class AiderHarness(CodingHarness):
    """
    Execute a TaskBatch through one Aider Worker session.

    The Worker is strictly confined to WORKSPACE_ROOT.

    LangGraph knows nothing about how edits happen.
    Aider owns prompting, tool use, and file editing.
    """

    def __init__(self, config: AiderConfig | None = None):
        self.config = config or AiderConfig()

    def execute(self, batch: TaskBatch) -> list[TaskResult]:
        """
        Execute the complete TaskBatch in one Aider session.

        A single Aider subprocess handles the entire batch.

        Files are passed to Aider anchored within WORKSPACE_ROOT.
        """

        if not batch.tasks:
            raise ValueError("TaskBatch must contain at least one task.")

        if not batch.execution_order:
            raise ValueError("TaskBatch must define an execution_order.")

        if not WORKSPACE_ROOT.is_dir():
            raise FileNotFoundError(
                f"Worker workspace does not exist: {WORKSPACE_ROOT}"
            )

        # Ensure sandbox has its own git root or is treated as a clean root
        sandbox_git = WORKSPACE_ROOT / ".git"
        if not sandbox_git.exists():
            subprocess.run(
                ["git", "init", "-q"],
                cwd=WORKSPACE_ROOT,
                check=False,
            )

        tasks_by_id = {
            task.task_id: task
            for task in batch.tasks
        }

        ordered_tasks = []

        for task_id in batch.execution_order:
            task = tasks_by_id.get(task_id)

            if task is None:
                raise ValueError(
                    f"Task '{task_id}' appears in execution_order "
                    "but does not exist in batch.tasks."
                )

            ordered_tasks.append(task)

        files = sorted(
            {
                path
                for task in ordered_tasks
                for path in task.relevant_files
            }
            | set(batch.relevant_files)
        )

        if not files:
            raise ValueError(
                "TaskBatch must identify at least one relevant file."
            )

        # Validate every declared path before giving it to Aider.
        for path in files:
            self._safe_workspace_path(path)

        prompt = self._build_prompt(
            batch=batch,
            ordered_tasks=ordered_tasks,
        )

        before = self._snapshot_files(files)

        # Pass relative paths from within WORKSPACE_ROOT.
        # Files in TaskBatch are logical paths (e.g., "hello.py" or "src/module/file.py").
        # Aider runs with cwd=WORKSPACE_ROOT, so we pass paths relative to that.
        rel_files = [
            str(Path(path).relative_to("sandbox") if str(path).startswith("sandbox/") else Path(path))
            for path in files
        ]

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
            "--message",
            prompt,
            *rel_files,
        ]

        print("\n=== AIDER WORKER ===")
        print(f"Model: {self.config.model}")
        print(f"Batch: {batch.batch_id}")
        print(f"Workspace: {WORKSPACE_ROOT}")
        print(f"Files: {', '.join(files)}")

        # Ensure environment doesn't propagate parent git repo vars
        env = os.environ.copy()
        env.pop("GIT_DIR", None)
        env.pop("GIT_WORK_TREE", None)

        completed = subprocess.run(
            command,
            cwd=WORKSPACE_ROOT,
            env=env,
            text=True,
        )

        # Collect results
        after = self._snapshot_files(files)
        task_results = []

        for task in ordered_tasks:
            task_files = task.relevant_files
            changed = [
                f for f in task_files
                if before.get(f) != after.get(f)
            ]
            status = "completed" if changed else "no_change"
            task_results.append(TaskResult(
                task_id=task.task_id,
                status=status,
                summary="Aider completed the Worker session. Task acceptance is verified separately by the validator.",
                files_changed=changed,
                errors=[],
            ))

        return task_results

    def _safe_workspace_path(self, path: str) -> Path:
        """Resolve a path relative to the workspace root, preventing directory traversal."""
        requested = (WORKSPACE_ROOT / path).resolve()
        if not requested.is_relative_to(WORKSPACE_ROOT):
            raise ValueError(f"Path {path} escapes workspace root")
        return requested

    def _build_prompt(self, *, batch: TaskBatch, ordered_tasks: list) -> str:
        """Build the prompt for the Aider session."""
        lines = [
            f"# Batch: {batch.batch_id}",
            f"# Objective: {batch.objective}",
            "",
            "## Tasks (in execution order):",
        ]
        for task in ordered_tasks:
            lines.append(f"### Task {task.task_id}: {task.objective}")
            lines.append(f"Scope: {task.scope}")
            if task.constraints:
                lines.append("Constraints:")
                for c in task.constraints:
                    lines.append(f"  - {c}")
            lines.append("Acceptance Criteria:")
            for ac in task.acceptance_criteria:
                lines.append(f"  - {ac}")
            lines.append("")

        lines.append("## Shared Context:")
        for ctx in batch.shared_context:
            lines.append(f"  - {ctx}")
        lines.append("")

        lines.append("## Relevant Files:")
        for f in batch.relevant_files:
            lines.append(f"  - {f}")
        lines.append("")

        lines.append("## Constraints:")
        for c in batch.constraints:
            lines.append(f"  - {c}")
        lines.append("")

        lines.append("Execute each task in order. Make minimal, focused changes.")
        lines.append("Do not add extra code, comments, or formatting beyond what is specified.")
        return "\n".join(lines)

    def _snapshot_files(self, files: list[str]) -> dict[str, str]:
        """Capture current content of each file (empty string if not exists)."""
        snapshot = {}
        for f in files:
            path = self._safe_workspace_path(f)
            if path.exists():
                snapshot[f] = path.read_text()
            else:
                snapshot[f] = ""
        return snapshot

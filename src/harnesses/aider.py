from __future__ import annotations

import hashlib
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

        Files are passed to Aider as workspace-relative paths and
        Aider runs with WORKSPACE_ROOT as its working directory.
        """

        if not batch.tasks:
            raise ValueError("TaskBatch must contain at least one task.")

        if not batch.execution_order:
            raise ValueError("TaskBatch must define an execution_order.")

        if not WORKSPACE_ROOT.is_dir():
            raise FileNotFoundError(
                f"Worker workspace does not exist: {WORKSPACE_ROOT}"
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
        # This prevents the harness itself from escaping the workspace.
        for path in files:
            self._safe_workspace_path(path)

        prompt = self._build_prompt(
            batch=batch,
            ordered_tasks=ordered_tasks,
        )

        # Snapshot the actual declared files rather than relying on Git.
        #
        # This works for:
        # - tracked files
        # - untracked files
        # - files created by the Foreman before Aider starts
        before = self._snapshot_files(files)

        command = [
            "aider",
            "--model",
            self.config.model,
            "--edit-format",
            self.config.edit_format,
            "--yes",
            "--no-auto-commits",
            "--no-git",
            "--message",
            prompt,
            *[
                str(Path(path))
                for path in files
            ],
        ]

        print("\n=== AIDER WORKER ===")
        print(f"Model: {self.config.model}")
        print(f"Batch: {batch.batch_id}")
        print(f"Workspace: {WORKSPACE_ROOT}")
        print(f"Files: {', '.join(files)}")

        completed = subprocess.run(
            command,
            cwd=WORKSPACE_ROOT,
            text=True,
        )

        after = self._snapshot_files(files)
        changed = sorted(
            path
            for path in files
            if before.get(path) != after.get(path)
        )

        if completed.returncode == 0:
            return [
                TaskResult(
                    task_id=task.task_id,
                    status="completed",
                    summary=(
                        "Aider completed the Worker session. "
                        "Task acceptance is verified separately by the validator."
                    ),
                    files_changed=sorted(
                        set(changed) & set(task.relevant_files)
                    ),
                    errors=[],
                )
                for task in ordered_tasks
            ]

        error = f"aider exited with code {completed.returncode}"

        return [
            TaskResult(
                task_id=task.task_id,
                status="failure",
                summary="Aider failed to complete the Worker session.",
                files_changed=sorted(
                    set(changed) & set(task.relevant_files)
                ),
                errors=[error],
            )
            for task in ordered_tasks
        ]

    def _safe_workspace_path(self, path: str) -> Path:
        """
        Resolve a workspace-relative path and ensure it remains
        inside WORKSPACE_ROOT.
        """

        candidate = Path(path)

        if candidate.is_absolute():
            raise ValueError(
                f"Absolute paths are not allowed in Worker files: {path}"
            )

        resolved = (WORKSPACE_ROOT / candidate).resolve()

        try:
            resolved.relative_to(WORKSPACE_ROOT)
        except ValueError as exc:
            raise ValueError(
                f"Worker file escapes workspace: {path}"
            ) from exc

        return resolved

    def _snapshot_files(
        self,
        files: list[str],
    ) -> dict[str, tuple[str, bytes] | None]:
        """
        Snapshot the content of declared workspace files.

        None means the file does not exist.

        Content hashing is unnecessary here because these are
        intentionally small/declared Worker files; storing the
        bytes also makes the comparison deterministic.
        """

        snapshots: dict[str, tuple[str, bytes] | None] = {}

        for path in files:
            target = self._safe_workspace_path(path)

            if not target.is_file():
                snapshots[path] = None
                continue

            content = target.read_bytes()

            snapshots[path] = (
                hashlib.sha256(content).hexdigest(),
                content,
            )

        return snapshots

    def _build_prompt(
        self,
        batch: TaskBatch,
        ordered_tasks,
    ) -> str:
        """
        Build the complete Worker prompt for one TaskBatch.
        """

        task_sections: list[str] = []

        for index, task in enumerate(ordered_tasks, start=1):
            task_sections.append(
                f"""
TASK {index}
Task ID:
{task.task_id}

Objective:
{task.objective}

Scope:
{task.scope}

Relevant files:
{chr(10).join(task.relevant_files)}

Dependencies:
{chr(10).join(task.dependencies) if task.dependencies else "None"}

Constraints:
{chr(10).join(task.constraints)}

Acceptance criteria:
{chr(10).join(task.acceptance_criteria)}
""".strip()
            )

        return f"""
You are the Worker coding agent.

You are executing one complete TaskBatch inside the isolated Worker workspace.

IMPORTANT WORKSPACE RULES

1. Your working directory is the Worker workspace.
2. Only modify the supplied relevant files.
3. Do not access, inspect, create, or modify files outside the supplied relevant files.
4. Do not access the parent repository.
5. Do not redesign the architecture.
6. Do not introduce unrelated changes.
7. Complete as much of the batch as possible within this single session.
8. The external validator will verify the acceptance criteria after you finish.
9. Follow acceptance criteria literally. Do not improve, refactor, wrap, or redesign requested code.
10. If the task specifies exact file content, reproduce that content exactly.
11. Do not add functions, classes, comments, docstrings, or abstractions unless explicitly requested.

BATCH OBJECTIVE
{batch.objective}

SHARED CONTEXT
{chr(10).join(batch.shared_context) if batch.shared_context else "None"}

BATCH CONSTRAINTS
{chr(10).join(batch.constraints) if batch.constraints else "None"}

EXECUTION ORDER
{chr(10).join(batch.execution_order)}

TASKS
{chr(10).join(task_sections)}
""".strip()
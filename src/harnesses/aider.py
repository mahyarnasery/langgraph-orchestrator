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

        # Pass relative paths from within WORKSPACE_ROOT to prevent path traversal confusion
        rel_files = [
            str(Path(path).name if not str(path).startswith("sandbox/") else Path(str(path).removeprefix("sandbox/")).name)
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
from typing import TypedDict

from .schemas import Plan, Task, WorkerResult


class WorkflowState(TypedDict):
    """Compact shared state passed between orchestration layers."""

    project_goal: str
    architecture_context: str

    plan: Plan | None
    current_task: Task | None
    worker_result: WorkerResult | None

    phase: str

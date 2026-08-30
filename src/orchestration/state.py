from typing import TypedDict

from .schemas import (
    CycleResult,
    Plan,
    TaskBatch,
    TaskResult,
    WorkerProfile,
    ValidationResult,
    ForemanReview,
)


class WorkflowState(TypedDict):
    """State owned and transitioned by the LangGraph workflow."""

    project_goal: str
    architecture_context: str
    worker_profile: WorkerProfile

    previous_cycle: CycleResult | None

    plan: Plan | None
    task_batch: TaskBatch | None
    task_results: list[TaskResult]

    validation_result: ValidationResult | None
    foreman_review: ForemanReview | None

    outcome: str
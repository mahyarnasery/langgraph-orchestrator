from typing import TypedDict

from .schemas import (
    CycleResult,
    ForemanReview,
    Plan,
    TaskBatch,
    TaskResult,
    ValidationResult,
    WorkerProfile,
)


class WorkflowState(TypedDict):
    """State owned and transitioned by the LangGraph workflow."""

    # ------------------------------------------------------------------
    # Persistent cycle input
    # ------------------------------------------------------------------

    project_goal: str
    architecture_context: str
    worker_profile: WorkerProfile
    previous_cycle: CycleResult | None

    # ------------------------------------------------------------------
    # Current-cycle orchestration state
    # ------------------------------------------------------------------

    plan: Plan | None

    task_batches: list[TaskBatch]

    task_results: list[TaskResult]

    worker_attempts: dict[str, int]

    foreman_repair_attempts: int

    validation_result: ValidationResult | None

    foreman_review: ForemanReview | None

    rework_mode: str

    phase: str

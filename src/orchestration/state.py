from typing import TypedDict

from .schemas import (
    CycleResult,
    ForemanReview,
    Plan,
    TaskBatch,
    TaskResult,
    WorkerProfile,
    ValidationResult,
)

class WorkflowState(TypedDict):
    """
    Compact shared orchestration state.

    Planner receives:
        - project_goal
        - architecture_context
        - worker_profile
        - previous_cycle

    Foreman additionally receives:
        - current plan

    Worker receives only:
        - current TaskBatch
    """

    # Persistent project context
    project_goal: str
    architecture_context: str

    # Static execution profile
    worker_profile: WorkerProfile

    # Previous-cycle memory (None during Cycle 1)
    previous_cycle: CycleResult | None

    # Current cycle
    plan: Plan | None
    task_batches: list[TaskBatch]
    task_results: list[TaskResult]

    validation_result: ValidationResult | None

    foreman_review: ForemanReview | None
    review_iteration: int
    
    # Runtime phase marker
    phase: str
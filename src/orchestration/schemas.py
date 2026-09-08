from pydantic import BaseModel, Field
from typing import Literal


# ---------------------------------------------------------------------------
# Planner
# ---------------------------------------------------------------------------

class Plan(BaseModel):
    """High-level architectural plan produced by the Planner."""

    objective: str = Field(
        description="What the implementation should accomplish."
    )

    approach: str = Field(
        description="High-level implementation approach."
    )

    affected_areas: list[str] = Field(
        description="Repository files, modules, or components likely to be affected."
    )

    acceptance_criteria: list[str] = Field(
        description="Conditions that must be satisfied for the work to be considered complete."
    )


# ---------------------------------------------------------------------------
# Worker profile
# ---------------------------------------------------------------------------

class WorkerProfile(BaseModel):
    """
    Compact description of the Worker model's capabilities.

    This is planning metadata consumed by the Planner and Foreman.
    It is not runtime telemetry.
    """

    model: str = Field(
        description="Worker model identifier."
    )

    context_limit: str = Field(
        description="Qualitative context constraint (e.g. 'limited')."
    )

    strengths: list[str] = Field(
        description="Kinds of implementation work the Worker performs well."
    )

    avoid: list[str] = Field(
        description="Work the Worker should not be assigned."
    )

    execution_mode: str = Field(
        description="Execution strategy (e.g. one_session_per_batch)."
    )


# ---------------------------------------------------------------------------
# Task
# ---------------------------------------------------------------------------

class Task(BaseModel):
    """
    Concrete implementation task assigned to the Worker harness.
    """

    task_id: str = Field(
        description="Unique task identifier."
    )

    objective: str = Field(
        description="Specific implementation objective."
    )

    scope: str = Field(
        description="Explicit boundaries of what may be modified."
    )

    relevant_files: list[str] = Field(
        description="Files relevant to the task."
    )

    dependencies: list[str] = Field(
        default_factory=list,
        description="Task IDs that must complete before this task."
    )

    constraints: list[str] = Field(
        description="Technical or architectural constraints."
    )

    acceptance_criteria: list[str] = Field(
        description="Concrete completion conditions."
    )


# ---------------------------------------------------------------------------
# Task batching
# ---------------------------------------------------------------------------

class TaskBatch(BaseModel):
    """
    Group of related tasks executed during one Worker harness session.
    """

    batch_id: str = Field(
        description="Unique identifier for the batch."
    )

    objective: str = Field(
        description="Shared objective of the batch."
    )

    tasks: list[Task] = Field(
        description="Tasks included in the batch."
    )

    shared_context: list[str] = Field(
        default_factory=list,
        description="Compact context shared across every task."
    )

    relevant_files: list[str] = Field(
        default_factory=list,
        description="Union of repository files relevant to the batch."
    )

    constraints: list[str] = Field(
        default_factory=list,
        description="Batch-wide constraints."
    )

    execution_order: list[str] = Field(
        description="Ordered list of task IDs."
    )


# ---------------------------------------------------------------------------
# Worker / harness execution result
# ---------------------------------------------------------------------------

class TaskResult(BaseModel):
    """
    Compact result returned by the coding harness after task execution.

    The orchestration layer does not store individual tool calls.
    """

    task_id: str = Field(
        description="Completed task identifier."
    )

    status: Literal["completed", "partial", "failure"] = Field(
        description="Execution outcome of the task."
    )

    summary: str = Field(
        description="Concise summary of what was accomplished."
    )

    files_changed: list[str] = Field(
        default_factory=list,
        description="Files modified by the harness."
    )

    errors: list[str] = Field(
        default_factory=list,
        description="Execution or validation errors."
    )


# ---------------------------------------------------------------------------
# Foreman review
# ---------------------------------------------------------------------------

class ForemanReview(BaseModel):
    """
    Review and integration decision produced by the Foreman.
    """

    decision: str = Field(
        description="accept | repair | rebuild"
    )

    accepted: bool = Field(
        description="True only when the implementation is accepted."
    )

    reviewed_task_ids: list[str] = Field(
        default_factory=list,
        description="Task IDs evaluated during this review."
    )

    problems: list[str] = Field(
        default_factory=list,
        description="Concrete implementation problems identified."
    )

    repair_instructions: list[str] = Field(
        default_factory=list,
        description="Precise instructions for repairing or rebuilding the implementation."
    )

    rebuild_task_ids: list[str] = Field(
        default_factory=list,
        description="Task IDs that must be rebuilt from clean files."
    )

    summary: str = Field(
        description="Brief explanation of the review decision."
    )


# ---------------------------------------------------------------------------
# Cycle memory
# ---------------------------------------------------------------------------

class CycleResult(BaseModel):
    """
    Compact orchestration record persisted into the next cycle.

    The repository remains the source of truth for code.
    This object preserves execution decisions and outcomes.
    """

    cycle_id: int = Field(
        description="Sequential orchestration cycle number."
    )

    plan: Plan = Field(
        description="Planner output for this cycle."
    )

    task_batches: list[TaskBatch] = Field(
        description="Batches produced for Worker execution."
    )

    task_results: list[TaskResult] = Field(
        description="Coding harness execution results."
    )

    outcome: str = Field(
        description="overall_success | partial | failure"
    )


# ---------------------------------------------------------------------------
# Validator
# ---------------------------------------------------------------------------

class ValidationResult(BaseModel):
    """
    Deterministic validation produced by the Validator.

    This model contains only objective facts gathered from repository
    inspection. No LLM reasoning or subjective review belongs here.
    """

    passed: bool = Field(
        description="True only if every deterministic validation check passed."
    )

    checks: list[str] = Field(
        default_factory=list,
        description="Successful validation checks."
    )

    errors: list[str] = Field(
        default_factory=list,
        description="Deterministic validation failures."
    )

    files_checked: list[str] = Field(
        default_factory=list,
        description="Repository files inspected by the validator."
    )

    changed_files: list[str] = Field(
        default_factory=list,
        description="Files modified during the batch."
    )
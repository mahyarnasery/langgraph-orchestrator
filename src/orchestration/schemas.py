from pydantic import BaseModel, Field


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
# Foreman task
# ---------------------------------------------------------------------------

class Task(BaseModel):
    """
    A concrete executable task produced by the Foreman.

    Tasks are intentionally small and suitable for a limited-context Worker.
    """

    task_id: str = Field(
        description="Unique task identifier."
    )

    objective: str = Field(
        description="Specific implementation objective."
    )

    scope: str = Field(
        description="Explicit boundaries of what the Worker may modify."
    )

    relevant_files: list[str] = Field(
        description="Existing files the Worker may inspect or edit."
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
    Group of related tasks executed during one Worker model session.
    """

    batch_id: str = Field(
        description="Unique batch identifier."
    )

    objective: str = Field(
        description="Shared objective of the batch."
    )

    tasks: list[Task] = Field(
        description="Tasks included in this batch."
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
# Worker execution result
# ---------------------------------------------------------------------------

class ToolAction(BaseModel):
    """
    One observed filesystem tool invocation.

    This is evidence collected by the orchestration layer,
    not something the Worker invents.
    """

    tool: str = Field(
        description="Tool name (read_file, edit_file, ...)."
    )

    path: str | None = Field(
        default=None,
        description="Workspace path operated on, if applicable."
    )

    result: str = Field(
        description="Observed tool outcome (OK, EDIT_SUCCESS, TOOL_ERROR, ...)."
    )

class TaskResult(BaseModel):
    """
    Compact factual record of one completed task.

    Unlike Alpha01v1, this stores observed tool evidence instead of
    Worker-claimed actions.
    """

    task_id: str = Field(
        description="Completed task identifier."
    )

    status: str = Field(
        description="completed | partial | failure"
    )

    tool_actions: list[ToolAction] = Field(
        default_factory=list,
        description="Observed filesystem tool invocations."
    )

    files_changed: list[str] = Field(
        default_factory=list,
        description="Files successfully modified."
    )

    validation: list[str] = Field(
        default_factory=list,
        description="Checks performed by the orchestration layer."
    )

    errors: list[str] = Field(
        default_factory=list,
        description="Observed execution errors."
    )

class ForemanReview(BaseModel):
    """Implementation-level review and repair decision produced by the Foreman."""

    accepted: bool = Field(
        description="Whether the current implementation is acceptable."
    )

    reviewed_task_ids: list[str] = Field(
        description="Task IDs evaluated during this review."
    )

    problems: list[str] = Field(
        default_factory=list,
        description="Concrete implementation issues identified."
    )

    decision: str = Field(
        description=(
            "Action to take after review: "
            "accept, repair, or rebuild."
        )
    )

    repair_instructions: list[str] = Field(
        default_factory=list,
        description=(
            "Precise instructions for the Foreman when performing a "
            "direct repair or preparing a rebuilt task."
        )
    )

    rebuild_task_ids: list[str] = Field(
        default_factory=list,
        description=(
            "Task IDs that must be rebuilt and sent back to the Worker. "
            "Empty when no Worker rework is required."
        )
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
    This object preserves only execution decisions and outcomes.
    """

    cycle_id: int = Field(
        description="Sequential orchestration cycle number."
    )

    plan: Plan = Field(
        description="Planner output for this cycle."
    )

    task_batches: list[TaskBatch] = Field(
        description="Batches produced by the Foreman."
    )

    task_results: list[TaskResult] = Field(
        description="Worker execution results."
    )

    outcome: str = Field(
        description="overall_success | partial | failure"
    )

# ---------------------------------------------------------------------------
# Validator
# ---------------------------------------------------------------------------

class ValidationResult(BaseModel):
    """
    Deterministic validation produced by the Validator node.

    This model contains only objective facts gathered from repository
    inspection (syntax, file existence, scope compliance, etc.).
    No LLM reasoning or subjective review belongs here.
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
        description="Files modified by the Worker during this batch."
    )
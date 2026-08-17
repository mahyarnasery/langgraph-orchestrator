from pydantic import BaseModel, Field


class Plan(BaseModel):
    """High-level architectural plan produced by the planner."""

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


class Task(BaseModel):
    """
    A concrete executable task produced by the foreman.

    The task describes work that an existing-file worker can perform.
    Repository structure and file creation are intentionally outside
    the worker's authority.
    """

    task_id: str = Field(
        description="Unique identifier for this task."
    )

    description: str = Field(
        description="Specific implementation task for the worker."
    )

    scope: str = Field(
        description=(
            "Explicit boundaries of what the worker may and may not change."
        )
    )

    relevant_files: list[str] = Field(
        description=(
            "Existing files inside the worker workspace that the worker "
            "is authorized to inspect or modify."
        )
    )

    constraints: list[str] = Field(
        description=(
            "Technical, architectural, or repository constraints the "
            "worker must respect."
        )
    )

    acceptance_criteria: list[str] = Field(
        description="Concrete conditions the worker must satisfy."
    )


class WorkerResult(BaseModel):
    """Result returned by the worker after attempting a task."""

    task_id: str = Field(
        description="ID of the task that was attempted."
    )

    status: str = Field(
        description=(
            "Result status. Use values such as 'success', 'partial', "
            "or 'failure'."
        )
    )

    summary: str = Field(
        description="Concise description of what the worker actually did."
    )

    implementation: str = Field(
        description=(
            "Description of the code changes or patches actually produced."
        )
    )
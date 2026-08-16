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
        description="Files, modules, or components likely to be affected."
    )

    acceptance_criteria: list[str] = Field(
        description="Conditions that must be satisfied for the work to be considered complete."
    )


class Task(BaseModel):
    """A concrete executable task produced by the foreman."""

    task_id: str = Field(
        description="Unique identifier for this task."
    )

    description: str = Field(
        description="Specific implementation task for the worker."
    )

    scope: str = Field(
        description="Explicit boundaries of what the worker should and should not change."
    )

    relevant_files: list[str] = Field(
        description="Files relevant to this task."
    )

    constraints: list[str] = Field(
        description="Technical or architectural constraints the worker must respect."
    )

    acceptance_criteria: list[str] = Field(
        description="Conditions the worker must satisfy."
    )


class WorkerResult(BaseModel):
    """Result returned by the worker after attempting a task."""

    task_id: str = Field(
        description="ID of the task that was attempted."
    )

    status: str = Field(
        description="Result status, such as success or failure."
    )

    summary: str = Field(
        description="Concise description of what the worker did."
    )

    implementation: str = Field(
        description="Description of the implementation produced by the worker."
    )

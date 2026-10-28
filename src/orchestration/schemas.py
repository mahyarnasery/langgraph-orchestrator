from __future__ import annotations

from typing import Literal
from pydantic import BaseModel, Field, model_validator


# ---------------------------------------------------------------------------
# Execution Ledger (Concise Shared Context)
# ---------------------------------------------------------------------------

class LedgerEntry(BaseModel):
    """Single concise record of an orchestration step and its outcome."""

    cycle_id: int = Field(description="Cycle number.")
    step: int = Field(description="Sequential step number within the cycle.")
    layer: Literal[
        "planner", "foreman", "worker", "assembler", "validator"
    ] = Field(description="Which orchestration layer executed this step.")
    action: str = Field(description="Short action identifier.")
    target: str = Field(description="Target file, task_id, or symbol.")
    status: Literal["success", "warning", "failure"] = Field(
        description="Outcome status of the step."
    )
    detail: str = Field(
        description="Concise 1-line summary of what happened and why."
    )


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


class PlannerReview(BaseModel):
    """Final acceptance review produced by the Planner."""

    criteria_evaluation: list[str] = Field(
        default_factory=list,
        description="Step-by-step evaluation against each acceptance criterion (mark each PASS or FAIL)."
    )

    feedback: list[str] = Field(
        default_factory=list,
        description="Constructive architectural feedback or missing elements (empty if all criteria pass)."
    )

    summary: str = Field(
        description="Concise summary of final assessment."
    )

    accepted: bool = Field(
        description="MUST be True if all acceptance criteria pass in criteria_evaluation; False only if a criterion failed."
    )

    @model_validator(mode="after")
    def _sync_accepted_with_evaluation(self) -> "PlannerReview":
        """If the LLM evaluated all criteria as passing and reported no failures, ensure accepted is True."""
        if not self.accepted and self.criteria_evaluation:
            eval_text = " ".join(self.criteria_evaluation).lower()
            summary_lower = self.summary.lower()
            has_pass = any(w in eval_text for w in ("pass", "passed", "accepted", "satisfied", "met"))
            has_fail = any(
                w in eval_text
                for w in ("fail", "failed", "missing", "incorrect", "not met", "error:")
            )
            summary_confirms = any(
                phrase in summary_lower
                for phrase in (
                    "all acceptance criteria are met",
                    "meets all stated requirements",
                    "meets all specified requirements",
                    "satisfies all",
                    "fully functional",
                )
            )
            if has_pass and not has_fail and summary_confirms:
                self.accepted = True
        return self


# ---------------------------------------------------------------------------
# Worker profile
# ---------------------------------------------------------------------------

class WorkerProfile(BaseModel):
    """Compact description of the Worker model's capabilities."""

    model: str = Field(description="Worker model identifier.")
    context_limit: str = Field(description="Qualitative context constraint.")
    strengths: list[str] = Field(description="Work the Worker performs well.")
    avoid: list[str] = Field(description="Work the Worker should not be assigned.")
    execution_mode: str = Field(description="Execution strategy.")


# ---------------------------------------------------------------------------
# Task & Batch
# ---------------------------------------------------------------------------

class Task(BaseModel):
    """Concrete implementation task assigned to the Worker harness."""

    task_id: str = Field(description="Unique task identifier (e.g. 't1').")

    objective: str = Field(description="Specific implementation objective.")

    target_symbol: str | None = Field(
        default=None,
        description="Optional single function or class name this task implements (e.g. 'parse_scores')."
    )

    scope: str = Field(description="Explicit boundaries of what may be modified.")

    relevant_files: list[str] = Field(
        description="Target file(s) where this task's code belongs."
    )

    dependencies: list[str] = Field(
        default_factory=list,
        description="Task IDs that must complete before this task."
    )

    constraints: list[str] = Field(
        default_factory=list,
        description="Technical or architectural constraints."
    )

    acceptance_criteria: list[str] = Field(
        description="Concrete completion conditions."
    )


class TaskBatch(BaseModel):
    """Group of related tasks executed during a Worker harness phase."""

    batch_id: str = Field(description="Unique identifier for the batch.")
    objective: str = Field(description="Shared objective of the batch.")
    tasks: list[Task] = Field(description="Tasks included in the batch.")
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
    verification_command: str | None = Field(
        default=None,
        description="Optional command to verify batch behavior."
    )


# ---------------------------------------------------------------------------
# Worker / harness execution result
# ---------------------------------------------------------------------------

class TaskResult(BaseModel):
    """Compact result returned by the coding harness after task execution."""

    task_id: str = Field(description="Completed task identifier.")
    status: Literal["completed", "partial", "failure", "no_change"] = Field(
        description="Execution outcome of the task."
    )
    summary: str = Field(description="Concise summary of what was accomplished.")
    files_changed: list[str] = Field(
        default_factory=list,
        description="Logical files modified or staged by the harness."
    )
    staged_file: str | None = Field(
        default=None,
        description="Path to isolated staging snippet if task was staged."
    )
    errors: list[str] = Field(
        default_factory=list,
        description="Execution or validation errors."
    )


# ---------------------------------------------------------------------------
# Foreman Assembly, Review & Self-Repair
# ---------------------------------------------------------------------------

class AssembledModule(BaseModel):
    """Output produced when Foreman assembles multiple staged Worker snippets into a module."""

    path: str = Field(description="Target module file path (e.g. 'analytics.py').")
    assembled_code: str = Field(
        description="Complete, coherent Python module code combining all task snippets."
    )
    notes: str = Field(
        default="",
        description="Brief note on how snippets were integrated or adjusted."
    )


class DirectRepairPatch(BaseModel):
    """Specific localized file fix performed directly by Foreman."""

    path: str = Field(description="Workspace relative file path to patch.")
    action: Literal["overwrite", "edit"] = Field(
        default="overwrite",
        description="overwrite entire content or replace target text."
    )
    content: str = Field(
        default="",
        description="Full content to write if action is overwrite."
    )
    target_text: str = Field(
        default="",
        description="Existing text snippet to replace if action is edit."
    )
    replacement_text: str = Field(
        default="",
        description="Replacement text snippet if action is edit."
    )


class ForemanReview(BaseModel):
    """Review and integration decision produced by the Foreman."""

    decision: Literal["accept", "repair", "rebuild"] = Field(
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
    direct_repairs: list[DirectRepairPatch] = Field(
        default_factory=list,
        description="Concrete patches if Foreman decides to repair directly."
    )
    rebuild_task_ids: list[str] = Field(
        default_factory=list,
        description="Task IDs that must be rebuilt."
    )
    rollback_files: list[str] = Field(
        default_factory=list,
        description="Files to revert to pre-batch snapshot before rebuilding."
    )
    summary: str = Field(
        description="Brief explanation of the review decision."
    )

    @model_validator(mode="after")
    def _clean_empty_list_items(self) -> "ForemanReview":
        self.rollback_files = [f.strip() for f in self.rollback_files if f and f.strip()]
        self.rebuild_task_ids = [t.strip() for t in self.rebuild_task_ids if t and t.strip()]
        return self


# ---------------------------------------------------------------------------
# Cycle memory & Validator
# ---------------------------------------------------------------------------

class CycleResult(BaseModel):
    """Compact orchestration record persisted into the next cycle."""

    cycle_id: int = Field(description="Sequential orchestration cycle number.")
    plan: Plan = Field(description="Planner output for this cycle.")
    task_batches: list[TaskBatch] = Field(description="Batches produced for Worker execution.")
    task_results: list[TaskResult] = Field(description="Coding harness execution results.")
    ledger: list[LedgerEntry] = Field(
        default_factory=list,
        description="Concise chronological ledger of steps taken during this cycle."
    )
    planner_review: PlannerReview | None = Field(
        default=None,
        description="Planner final review result."
    )
    outcome: str = Field(description="overall_success | partial | failure")


class ValidationResult(BaseModel):
    """Deterministic validation produced by the Validator."""

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

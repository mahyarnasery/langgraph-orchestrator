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
        "planner", "foreman", "worker", "assembler", "validator", "checkpoint"
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
# Planner & Multi-Stage Blueprint Decomposition
# ---------------------------------------------------------------------------

class StageGoal(BaseModel):
    """Single sequential architectural milestone in a multi-stage project."""

    stage_id: int = Field(description="Sequential 1-based stage number (1, 2, ...).")
    title: str = Field(description="Concise title for this architectural stage.")
    objective: str = Field(
        description=(
            "Complete implementation specification for this stage, preserving exact "
            "file paths, class/function/method signatures, formulas, and expected outputs."
        )
    )
    target_files: list[str] = Field(
        default_factory=list,
        description="Relative workspace file paths created or updated in this stage.",
    )


class MultiStageBlueprint(BaseModel):
    """Ordered sequence of architectural stages produced by the Planner."""

    project_summary: str = Field(
        description="High-level summary of the overall multi-stage application."
    )
    stages: list[StageGoal] = Field(
        description="Sequential stages ordered from foundational models/utilities to services and entry points."
    )


class SymbolSpec(BaseModel):
    """Architectural contract for a single function or Class.method designed by the Planner."""

    target_symbol: str = Field(
        description=(
            "Exact function name (e.g. 'load_dam_profile') or Class.method name "
            "(e.g. 'TelemetryRecord.__init__', 'TelemetryRecord.is_valid')."
        )
    )
    signature: str = Field(
        description="Exact Python def signature (e.g. 'def is_valid(self) -> bool:')."
    )
    contract: str = Field(
        description="Exact behavior, formula, or return expression required for this symbol."
    )


class FileBlueprint(BaseModel):
    """Per-file architectural blueprint produced by the Planner."""

    path: str = Field(
        description="Relative workspace file path (e.g. 'config/dam_profile.json', 'domain/models.py', 'main.py')."
    )
    role: str = Field(
        description="Concise description of this file's purpose or exact data/config content."
    )
    required_imports: list[str] = Field(
        default_factory=list,
        description="Exact import lines required at the top of this Python file (empty for non-Python files)."
    )
    symbols: list[SymbolSpec] = Field(
        default_factory=list,
        description=(
            "Every standalone function or Class.method to implement in this file. "
            "For a class with 2+ methods, list each method separately as 'ClassName.method_name'. "
            "Leave empty for non-Python assets (.json, .csv) or simple top-level entry scripts (main.py)."
        ),
    )


class Plan(BaseModel):
    """Architectural blueprint produced by the Planner."""

    objective: str = Field(
        description="What the implementation should accomplish."
    )

    approach: str = Field(
        description="High-level implementation approach and dependency flow."
    )

    affected_areas: list[str] = Field(
        description="Repository files, modules, or components to be created or modified."
    )

    file_blueprints: list[FileBlueprint] = Field(
        default_factory=list,
        description=(
            "Detailed per-file architectural blueprints defining exact paths, imports, "
            "and function/method specifications (SymbolSpec) for the Foreman to dispatch."
        ),
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
        description=(
            "Single function name (e.g. 'parse_scores') or Class.method name "
            "(e.g. 'Account.__init__', 'Account.deposit') this task implements."
        ),
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

    test_assertions: list[str] = Field(
        default_factory=list,
        description=(
            "Executable Python assert statements using concrete literal inputs to verify "
            "this task's function or class method (e.g. [\"acc = Account('Alice', 100); assert acc.deposit(50) == 150\"]). "
            "Leave empty for pure entry-point scripts that only print output."
        ),
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
    test_assertions: list[str] = Field(
        default_factory=list,
        description=(
            "Optional module/batch-level Python assert statements verifying integrated "
            "cross-function or cross-module behavior."
        ),
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

class AtomicCodeSnippet(BaseModel):
    """Focused code output produced when Foreman (8B) escalates a failed 3B Worker task."""

    target_symbol: str | None = Field(
        default=None,
        description="Function or Class.method implemented.",
    )
    code: str = Field(
        description="Valid raw Python code (or JSON/CSV content) for this task without markdown fences."
    )


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
# Cycle memory, Checkpoint & Validator
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
    failed_task_ids: list[str] = Field(
        default_factory=list,
        description="Task IDs that failed deterministic symbol, assertion, or acceptance checks."
    )


class CheckpointData(BaseModel):
    """Persistent orchestration state saved to sandbox/.checkpoint/state.json."""

    cycle_id: int = Field(default=1)
    project_goal: str = Field(default="")
    architecture_context: str = Field(default="")
    worker_profile: WorkerProfile
    previous_cycle: CycleResult | None = Field(default=None)
    plan: Plan | None = Field(default=None)
    task_batches: list[TaskBatch] = Field(default_factory=list)
    task_results: list[TaskResult] = Field(default_factory=list)
    ledger: list[LedgerEntry] = Field(default_factory=list)
    staged_files: dict[str, dict[str, str]] = Field(default_factory=dict)
    file_snapshots: dict[str, str] = Field(default_factory=dict)
    worker_attempts: dict[str, int] = Field(default_factory=dict)
    foreman_repair_attempts: int = Field(default=0)
    rebuild_attempts: int = Field(default=0)
    validation_result: ValidationResult | None = Field(default=None)
    foreman_review: ForemanReview | None = Field(default=None)
    planner_review: PlannerReview | None = Field(default=None)
    rework_mode: str = Field(default="none")
    phase: str = Field(default="initial")
    completed_cycles: list[CycleResult] = Field(default_factory=list)
    stages: list[StageGoal] = Field(default_factory=list)
    current_stage_index: int = Field(default=0)
    completed_stage_ids: list[int] = Field(default_factory=list)

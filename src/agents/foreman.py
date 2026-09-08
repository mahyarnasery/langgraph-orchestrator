from src.orchestration.models import FOREMAN_MODEL, model_session
from src.orchestration.schemas import (
    CycleResult,
    ForemanReview,
    Plan,
    TaskBatch,
    TaskResult,
    ValidationResult,
    WorkerProfile,
)


class ForemanAgent:
    """Implementation coordination, review, and integration agent."""

    def create_batch(
        self,
        *,
        project_goal: str,
        architecture_context: str,
        worker_profile: WorkerProfile,
        previous_cycle: CycleResult | None,
        plan: Plan,
    ) -> TaskBatch:
        worker_profile_text = worker_profile.model_dump_json(
            indent=2
        )

        previous_cycle_text = (
            previous_cycle.model_dump_json(indent=2)
            if previous_cycle is not None
            else "None (Cycle 1)"
        )

        # Include previous cycle execution evidence if available
        previous_evidence = ""
        if previous_cycle is not None:
            previous_evidence = f"""
Previous Cycle Execution Evidence:
- Plan: {previous_cycle.plan.objective}
- Outcome: {previous_cycle.outcome}
- Task Results: {len(previous_cycle.task_results)} tasks executed
- Planner Review: {'Accepted' if previous_cycle.planner_review and previous_cycle.planner_review.accepted else 'Rejected/Revision Needed'}
"""

        prompt = f"""
You are the Foreman / Integrator.

Translate the Planner's architectural plan into one executable TaskBatch.

Project goal:
{project_goal}

Architecture context:
{architecture_context}

Worker profile:
{worker_profile_text}

Previous cycle:
{previous_cycle_text}
{previous_evidence}

Planner plan:
{plan.model_dump_json(indent=2)}

Responsibilities:
- Decompose the plan into concrete implementation tasks.
- Define task dependencies.
- Define task scope.
- Define relevant files.
- Define acceptance criteria.
- Keep tasks small enough for the Worker.
- Do not redesign the Planner's architecture.
- Use previous cycle evidence to avoid repeating failures.

The Worker edits existing files only.
If a required file does not exist, the orchestration layer will prepare it.

Every file mentioned by a task must appear in TaskBatch.relevant_files.

Return only the structured TaskBatch.
"""

        with model_session(FOREMAN_MODEL) as llm:
            structured = llm.with_structured_output(TaskBatch)
            return structured.invoke(prompt)

    def create_corrective_batch(
        self,
        *,
        batch: TaskBatch,
        review: ForemanReview,
        task_results: list[TaskResult],
    ) -> TaskBatch:
        """Create a targeted corrective TaskBatch to rebuild/fix failed tasks."""

        prompt = f"""
You are the Foreman creating a corrective TaskBatch for the Worker.

Original batch:
{batch.model_dump_json(indent=2)}

Foreman review and rebuild instructions:
{review.model_dump_json(indent=2)}

Previous task execution outcomes:
{', '.join(f'{r.task_id}: {r.status}' for r in task_results)}

Generate a new, focused TaskBatch addressing only the tasks needing rebuild/correction.
All tasks must specify unambiguous scope, constraints, and acceptance criteria.
Ensure every referenced file is listed in relevant_files.

Return only the structured TaskBatch.
"""

        with model_session(FOREMAN_MODEL) as llm:
            structured = llm.with_structured_output(TaskBatch)
            return structured.invoke(prompt)

    def review(
        self,
        *,
        project_goal: str,
        architecture_context: str,
        plan: Plan,
        batch: TaskBatch,
        task_results: list[TaskResult],
        validation_result: ValidationResult,
        repository_context: str,
        worker_attempts: dict[str, int],
    ) -> ForemanReview:
        """Review the Worker implementation and choose the next action."""

        task_results_context = "\n\n".join(
            result.model_dump_json(indent=2)
            for result in task_results
        )

        validation_context = validation_result.model_dump_json(
            indent=2
        )

        worker_attempts_context = "\n".join(
            f"{task_id}: {attempts}"
            for task_id, attempts in sorted(
                worker_attempts.items()
            )
        )

        prompt = f"""
You are the Foreman / Integrator performing an implementation review.

The Planner owns architecture.
You own implementation-level integration.

Project goal:
{project_goal}

Architecture context:
{architecture_context}

Planner plan:
{plan.model_dump_json(indent=2)}

Current TaskBatch:
{batch.model_dump_json(indent=2)}

Worker results:
{task_results_context}

Deterministic validation:
{validation_context}

Actual repository contents:
{repository_context}

Worker attempt counts:
{worker_attempts_context}

Determine the smallest appropriate next action.

Choose exactly ONE decision:

"accept"
    The implementation satisfies the Planner's acceptance criteria
    and has no meaningful implementation problem.

"repair"
    The implementation is fundamentally correct but has a small,
    localized problem that you (Foreman) can repair directly yourself.
    When choosing "repair", you MUST provide direct_repairs with the exact
    file path and changes (overwrite or edit).

"rebuild"
    One or more Worker tasks should be sent back to Worker because the
    implementation needs substantial changes, logic rewrite, or additions.

Rules:
1. Do not redesign the architecture.
2. The deterministic Validator is authoritative for syntax, file existence, and scope checks.
3. Actual repository contents are the source of truth.
4. Prefer "repair" when the fix is small/minor (e.g. typos, single missing print/return, small syntax tweak).
5. For "repair", populate direct_repairs with precise actions.
6. For "rebuild", populate rebuild_task_ids and repair_instructions.
7. For "accept", accepted must be true, and problems, repair_instructions, direct_repairs, and rebuild_task_ids must be empty.

Return only the structured ForemanReview.
"""

        with model_session(FOREMAN_MODEL) as llm:
            structured = llm.with_structured_output(ForemanReview)
            return structured.invoke(prompt)

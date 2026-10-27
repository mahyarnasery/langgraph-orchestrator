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
        worker_profile_text = worker_profile.model_dump_json(indent=2)

        previous_cycle_text = (
            previous_cycle.model_dump_json(indent=2)
            if previous_cycle is not None
            else "None (Cycle 1)"
        )

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
- CRITICAL: The orchestration layer automatically pre-creates all empty files before the Worker runs. NEVER create a separate task just to create an empty file. Every task must write or modify actual code.
- If a single file only needs one small implementation, use a SINGLE task for that file.
- For multi-file plans, create ONE task per file in dependency order (e.g., Task 1 creates the helper/library module, Task 2 creates the entry-point script with `dependencies: ["t1"]`).
- CRITICAL: Each task's `relevant_files` must list ONLY the file(s) that specific task writes or modifies. Do NOT list imported dependency files in `task.relevant_files` (the harness automatically passes dependency files as read-only context).
- Define task dependencies and populate execution_order with all task_ids.
- Define task scope, relevant_files, and concrete acceptance_criteria.
- Leave verification_command as null unless a specific pytest/unittest suite must be executed (NEVER use shell pipes like `|` or `grep` in verification_command; the Validator already checks syntax, file contents, and executes Python scripts automatically).
- Do not redesign the Planner's architecture.

Every file mentioned by a task must appear in both task.relevant_files and TaskBatch.relevant_files.

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
- Populate execution_order with the exact task_ids in this corrective batch.
- Specify unambiguous scope, constraints, and acceptance criteria.
- Leave verification_command as null unless running a dedicated test runner without shell pipes.
- Ensure every referenced file is listed in both task.relevant_files and TaskBatch.relevant_files.

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
        foreman_repair_attempts: int = 0,
        max_foreman_repairs: int = 2,
    ) -> ForemanReview:
        """Review the Worker implementation and choose the next action."""

        task_results_context = "\n\n".join(
            result.model_dump_json(indent=2) for result in task_results
        )

        validation_context = validation_result.model_dump_json(indent=2)

        worker_attempts_context = "\n".join(
            f"{task_id}: {attempts}"
            for task_id, attempts in sorted(worker_attempts.items())
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

Foreman direct self-repair attempts used: {foreman_repair_attempts} / {max_foreman_repairs}
{"CRITICAL: You have exhausted all direct self-repair attempts. Do NOT choose 'repair'. Choose 'accept' (if the code works) or 'rebuild'." if foreman_repair_attempts >= max_foreman_repairs else ""}

Determine the smallest appropriate next action.

Choose exactly ONE decision:

"accept"
    The implementation satisfies the Planner's acceptance criteria
    and passes deterministic validation with no meaningful problem.

"repair"
    The implementation is fundamentally close or small enough that you
    (Foreman) can repair it directly yourself.
    When choosing "repair", you MUST provide direct_repairs with the exact
    file path and changes (prefer action="overwrite" with full file content
    for small files, or action="edit" with exact target_text/replacement_text).

"rebuild"
    One or more Worker tasks should be sent back to Worker because the
    implementation needs substantial changes, logic rewrite, or additions.
    If the Worker corrupted a file and it should be reset before rebuilding,
    list those paths in rollback_files.

Rules:
1. Do not redesign the architecture.
2. The deterministic Validator is authoritative for syntax, file existence, and scope checks.
3. Actual repository contents are the source of truth.
4. Prefer "repair" when the fix is small/minor (e.g. typos, single missing print/return, small syntax tweak, or empty file needing a 1-5 line implementation).
5. For "repair", populate direct_repairs with precise actions.
6. For "rebuild", populate rebuild_task_ids and repair_instructions (and rollback_files if corrupted).
7. For "accept", accepted must be true, and problems, repair_instructions, direct_repairs, and rebuild_task_ids must be empty.

Return only the structured ForemanReview.
"""

        with model_session(FOREMAN_MODEL) as llm:
            structured = llm.with_structured_output(ForemanReview)
            return structured.invoke(prompt)

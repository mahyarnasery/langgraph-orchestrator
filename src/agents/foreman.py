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
    """Implementation coordination and integration agent."""

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
            else "None"
        )

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

The Worker edits existing files only.
If a required file does not exist, the orchestration layer will prepare it.

Every file mentioned by a task must appear in TaskBatch.relevant_files.

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
    localized problem that the Foreman can repair directly.

"rebuild"
    One or more Worker tasks should be rebuilt from clean files because
    the existing implementation is substantially wrong, corrupted,
    incomplete, or unreliable to patch.

Rules:

1. Do not redesign the architecture.

2. The deterministic Validator is authoritative for:
   - syntax
   - file existence
   - deterministic scope checks

3. The actual repository contents are the source of truth for
   implementation correctness.

4. Worker results are execution evidence, not proof of correctness.

5. Every reported problem must be concrete.

6. Identify the affected file and function or implementation area
   whenever possible.

7. Prefer "repair" over "rebuild" when the problem is small and localized.

8. Prefer "rebuild" when patching the existing implementation would be
   less reliable than starting from a clean file.

9. If a task has already reached three Worker attempts, do not request
   another Worker attempt for that task. Use "repair" if the Foreman can
   fix it directly. Otherwise report the problem clearly.

10. For "accept":
    - accepted must be true
    - problems must be empty
    - repair_instructions must be empty
    - rebuild_task_ids must be empty

11. For "repair":
    - accepted must be false
    - provide precise repair_instructions
    - rebuild_task_ids must be empty

12. For "rebuild":
    - accepted must be false
    - provide precise repair_instructions explaining the failure
    - rebuild_task_ids must contain the task IDs requiring rebuild

13. Do not invent a TaskBatch during review.

14. Return only the structured ForemanReview.
"""

        with model_session(FOREMAN_MODEL) as llm:
            structured = llm.with_structured_output(ForemanReview)
            return structured.invoke(prompt)
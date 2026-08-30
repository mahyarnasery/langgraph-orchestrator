from src.orchestration.models import FOREMAN_MODEL, model_session
from src.orchestration.schemas import (
    ForemanReview,
    Plan,
    TaskBatch,
)


class ForemanAgent:
    """Implementation coordination and integration agent."""

    def create_batch(
        self,
        *,
        project_goal: str,
        architecture_context: str,
        worker_profile: str,
        previous_cycle: str,
        plan: Plan,
    ) -> TaskBatch:
        prompt = f"""
You are the Foreman / Integrator.

Translate the Planner's architectural plan into one executable TaskBatch.

Project goal:
{project_goal}

Architecture context:
{architecture_context}

Worker profile:
{worker_profile}

Previous cycle:
{previous_cycle}

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
        task_results: str,
        validation_result: str,
        repository_context: str,
        worker_attempts: str,
    ) -> ForemanReview:
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
{task_results}

Deterministic validation:
{validation_result}

Actual repository contents:
{repository_context}

Worker attempt counts:
{worker_attempts}

Determine the smallest appropriate next action.

Choose exactly one:

accept
    Implementation is correct.

repair
    Implementation is fundamentally correct but requires a small localized
    Foreman repair.

rebuild
    A Worker task should be rebuilt from a clean file.

Rules:
- Do not redesign architecture.
- Validator evidence is authoritative for deterministic checks.
- Repository contents are the source of truth.
- Worker results are execution evidence, not proof of correctness.
- Every problem must be concrete.
- Prefer repair over rebuild for small problems.
- Prefer rebuild over repeatedly patching a fundamentally bad implementation.
- If a task has already failed three Worker attempts, do not request it again.

Return only the structured ForemanReview.
"""

        with model_session(FOREMAN_MODEL) as llm:
            structured = llm.with_structured_output(ForemanReview)
            return structured.invoke(prompt)
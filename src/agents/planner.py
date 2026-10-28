from src.orchestration.models import PLANNER_MODEL, model_session
from src.orchestration.schemas import (
    CycleResult,
    Plan,
    PlannerReview,
    ValidationResult,
    WorkerProfile,
)


class PlannerAgent:
    """High-level planning and acceptance agent."""

    def run(
        self,
        *,
        project_goal: str,
        architecture_context: str,
        worker_profile: WorkerProfile,
        previous_cycle: CycleResult | None,
        ledger_context: str = "",
    ) -> Plan:
        previous_cycle_text = (
            previous_cycle.model_dump_json(indent=2)
            if previous_cycle is not None
            else "None (Cycle 1)"
        )

        worker_profile_text = worker_profile.model_dump_json(indent=2)

        prompt = f"""
You are the Planner / Architect.

Responsibilities:
- Analyze the project goal.
- Define the high-level implementation approach.
- Identify affected repository areas.
- Define clear acceptance criteria.
- Do not implement code.

Project goal:
{project_goal}

Architecture context:
{architecture_context}

Execution Ledger:
{ledger_context}

Worker profile:
{worker_profile_text}

Previous cycle:
{previous_cycle_text}


Return only the structured Plan.
"""

        with model_session(PLANNER_MODEL) as llm:
            structured = llm.with_structured_output(Plan)
            return structured.invoke(prompt)

    def review(
        self,
        *,
        project_goal: str,
        plan: Plan,
        repository_context: str,
        validation_result: ValidationResult | None = None,
        ledger_context: str = "",
    ) -> PlannerReview:
        """Evaluate whether the integrated repository code satisfies the blueprint."""

        validation_text = (
            validation_result.model_dump_json(indent=2)
            if validation_result is not None
            else "Not available"
        )

        prompt = f"""
You are the Planner / Architect conducting the final acceptance review AFTER the Worker and Foreman have finished implementing and assembling the repository.

Project goal:
{project_goal}

Initial architectural blueprint & acceptance criteria:
{plan.model_dump_json(indent=2)}

Execution Ledger (steps taken & verified in this cycle):
{ledger_context}

Deterministic Validator results (post-implementation):
{validation_text}

Final integrated repository content (post-implementation state on disk):
{repository_context}

Instructions:
1. Evaluate each acceptance criterion against the final repository content and Validator results.
2. Note that the Worker and Foreman have ALREADY applied and assembled the changes shown above.
3. If every acceptance criterion is satisfied by the current repository content and deterministic validation passed, you MUST set `accepted` to `true`.
4. Only set `accepted` to `false` if a required file, function, or behavior is still missing or broken in the final repository content.


Return only the structured PlannerReview.
"""

        with model_session(PLANNER_MODEL) as llm:
            structured = llm.with_structured_output(PlannerReview)
            return structured.invoke(prompt)

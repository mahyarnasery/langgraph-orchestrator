from src.orchestration.models import PLANNER_MODEL, model_session
from src.orchestration.schemas import CycleResult, Plan, PlannerReview, WorkerProfile


class PlannerAgent:
    """High-level planning and acceptance agent."""

    def run(
        self,
        *,
        project_goal: str,
        architecture_context: str,
        worker_profile: WorkerProfile,
        previous_cycle: CycleResult | None,
    ) -> Plan:
        previous_cycle_text = (
            previous_cycle.model_dump_json(indent=2)
            if previous_cycle is not None
            else "None (Cycle 1)"
        )

        worker_profile_text = worker_profile.model_dump_json(
            indent=2
        )

        prompt = f"""
You are the Planner / Architect.

Responsibilities:
- Analyze the project goal.
- Define the high-level implementation approach.
- Identify affected repository areas.
- Define acceptance criteria.
- Do not implement code.
- Do not modify the repository.

Project goal:
{project_goal}

Architecture context:
{architecture_context}

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
    ) -> PlannerReview:
        """Evaluate whether the integrated repository code satisfies the blueprint."""

        prompt = f"""
You are the Planner / Architect conducting the final acceptance review.

Project goal:
{project_goal}

Initial architectural blueprint & acceptance criteria:
{plan.model_dump_json(indent=2)}

Final integrated repository content:
{repository_context}

Evaluate each acceptance criterion against the actual repository content.
Determine whether the overall solution is accepted.

Return only the structured PlannerReview.
"""

        with model_session(PLANNER_MODEL) as llm:
            structured = llm.with_structured_output(PlannerReview)
            return structured.invoke(prompt)

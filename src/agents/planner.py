from src.orchestration.models import PLANNER_MODEL, model_session
from src.orchestration.schemas import Plan


class PlannerAgent:
    """High-level planning agent."""

    def run(
        self,
        *,
        project_goal: str,
        architecture_context: str,
        worker_profile: str,
        previous_cycle: str,
    ) -> Plan:
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
{worker_profile}

Previous cycle:
{previous_cycle}

Return only the structured Plan.
"""

        with model_session(PLANNER_MODEL) as llm:
            structured = llm.with_structured_output(Plan)
            return structured.invoke(prompt)
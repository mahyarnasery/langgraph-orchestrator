from src.orchestration.models import get_planner_model, model_session
from src.orchestration.schemas import (
    CycleResult,
    MultiStageBlueprint,
    Plan,
    PlannerReview,
    StageGoal,
    ValidationResult,
    WorkerProfile,
)


def _summarize_previous_cycle(previous_cycle: CycleResult | None) -> str:
    """Build a compact summary of the previous cycle without dumping raw nested JSON."""
    if previous_cycle is None:
        return "None (First cycle for this goal)"

    lines = [
        f"Cycle {previous_cycle.cycle_id} outcome: {previous_cycle.outcome}",
    ]
    if previous_cycle.planner_review:
        lines.append(f"Review summary: {previous_cycle.planner_review.summary}")
        if previous_cycle.planner_review.feedback:
            lines.append(
                "Remaining feedback: " + "; ".join(previous_cycle.planner_review.feedback[:4])
            )
    return "\n".join(lines)


class PlannerAgent:
    """High-level planning, multi-stage decomposition, and acceptance agent."""

    def decompose_stages(
        self,
        *,
        project_goal: str,
        architecture_context: str,
        max_stages: int = 3,
    ) -> MultiStageBlueprint:
        """
        Decompose a larger project specification into sequential architectural stages
        (e.g. Stage 1: Config/data & core models -> Stage 2: Service layer & entry point).
        """
        prompt = f"""
You are the Planner / Architect decomposing a project into at most {max_stages} sequential build stages.

Project goal:
{project_goal}

Architecture context:
{architecture_context}

Decomposition Rules (CRITICAL):
1. Order stages from foundational config/data files (`.json`, `.csv`, `.toml`) and core models/utilities up to higher-level services and entry-point scripts (`main.py` / `app.py`).
2. Group tightly coupled entry scripts (`main.py`) with the final service/application stage so every stage produces valid, testable files.
3. Preserve EXACT file paths, JSON/CSV keys, class names, method/function signatures, formulas, and expected print/return strings from the Project goal inside each stage's `objective`.
4. Do NOT invent extra files or stages not requested in the Project goal.


Return only the structured MultiStageBlueprint.
"""

        with model_session(get_planner_model()) as llm:
            structured = llm.with_structured_output(MultiStageBlueprint)
            blueprint = structured.invoke(prompt)

        if not blueprint.stages:
            return MultiStageBlueprint(
                project_summary=project_goal.strip().splitlines()[0],
                stages=[
                    StageGoal(
                        stage_id=1,
                        title="Full Project Implementation",
                        objective=project_goal,
                        target_files=[],
                    )
                ],
            )

        cleaned_stages: list[StageGoal] = []
        for idx, stg in enumerate(blueprint.stages[:max_stages], start=1):
            norm_files = [
                f.strip().removeprefix("sandbox/").strip("/")
                for f in stg.target_files
                if f and f.strip()
            ]
            cleaned_stages.append(
                StageGoal(
                    stage_id=idx,
                    title=stg.title.strip() or f"Stage {idx}",
                    objective=stg.objective.strip() or project_goal,
                    target_files=norm_files,
                )
            )

        return MultiStageBlueprint(
            project_summary=blueprint.project_summary,
            stages=cleaned_stages,
        )

    def run(
        self,
        *,
        project_goal: str,
        architecture_context: str,
        worker_profile: WorkerProfile,
        previous_cycle: CycleResult | None,
        ledger_context: str = "",
    ) -> Plan:
        previous_cycle_text = _summarize_previous_cycle(previous_cycle)
        worker_profile_text = worker_profile.model_dump_json(indent=2)

        prompt = f"""
You are the Planner / Architect.

Responsibilities:
- Analyze the project goal.
- Define the high-level implementation approach.
- Identify affected repository areas (including both Python `.py` modules and any non-Python config/data files such as `.json`, `.csv`, `.toml`, `.sql`, `.md`).
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

Previous cycle summary:
{previous_cycle_text}


Return only the structured Plan.
"""

        with model_session(get_planner_model()) as llm:
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
3. If every acceptance criterion for the active goal is satisfied by the current repository content and deterministic validation passed, you MUST set `accepted` to `true`.
4. Only set `accepted` to `false` if a required file, function, class, or behavior for the active goal is still missing or broken in the final repository content.


Return only the structured PlannerReview.
"""

        with model_session(get_planner_model()) as llm:
            structured = llm.with_structured_output(PlannerReview)
            return structured.invoke(prompt)

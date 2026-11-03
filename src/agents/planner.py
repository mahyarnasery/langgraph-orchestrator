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
        return "None (First cycle)"

    lines = [f"Cycle {previous_cycle.cycle_id} outcome: {previous_cycle.outcome}"]
    if previous_cycle.planner_review:
        lines.append(f"Review: {previous_cycle.planner_review.summary}")
        if previous_cycle.planner_review.feedback:
            lines.append("Feedback: " + "; ".join(previous_cycle.planner_review.feedback[:3]))
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
        prompt = f"""
You are the Lead Software Architect decomposing a project into at most {max_stages} sequential build stages.

Project goal:
{project_goal}

Architecture context:
{architecture_context}

Rules:
1. Order stages from foundational config/data (`.json`, `.csv`, `.toml`) and domain models up to services and `main.py`.
2. Populate `target_files` for each stage with exact relative file paths (e.g. `["config/dam_profile.json", "data/telemetry.csv", "domain/models.py"]`).
3. Keep each stage's `objective` to a concise 1-2 sentence summary (exact file specs are attached automatically).
4. Do NOT invent extra files not requested in the Project goal.

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

        prompt = f"""
You are the Lead Software Architect designing the `Plan` for the active stage.

Project goal:
{project_goal}

Repository & Architecture:
{architecture_context}

Previous cycle: {previous_cycle_text}

Blueprint Rules (CRITICAL):
1. Populate `affected_areas` and `file_blueprints` ONLY with files requested for this active stage. Do NOT include already-implemented files from earlier stages.
2. For non-Python files (`.json`, `.csv`, `.toml`, etc.) and entry scripts (`main.py`, `app.py`):
   - Describe required keys/rows/steps in `role`, and MUST leave `symbols` empty (`[]`).
3. For Python library/domain/service modules (`.py`):
   - Populate `required_imports` with exact import lines (e.g. `["from domain.models import TelemetryRecord"]`).
   - For standalone functions: set `target_symbol` to the function name (e.g. `"analyze_flow"`), `signature` to `"def analyze_flow(...)"`, and `contract` to a 1-line behavior description.
   - For classes: do NOT create a separate symbol for the class header. List each method with `target_symbol` formatted as `"ClassName.method_name"` (e.g. `"TelemetryRecord.__init__"`, `"TelemetryRecord.is_flood"`), `signature` as `"def __init__(self, ...)"`, and a non-empty `contract`.

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
        if validation_result is None:
            validation_text = "Not available"
        elif validation_result.passed:
            stdout_checks = [c for c in validation_result.checks if c.startswith("stdout (")]
            validation_text = (
                f"PASSED: All {len(validation_result.checks)} checks passed (0 errors). "
                f"Files checked: {', '.join(validation_result.files_checked)}. "
                + (" | ".join(stdout_checks) if stdout_checks else "")
            )
        else:
            validation_text = validation_result.model_dump_json(indent=2)

        prompt = f"""
You are the Planner conducting the final acceptance review.

Objective: {plan.objective}
Acceptance criteria:
{chr(10).join(f'- {c}' for c in plan.acceptance_criteria)}

Validator results:
{validation_text}

Repository digest:
{repository_context}

If deterministic validation passed and required files/symbols exist, set `accepted` to `true`.
Return only the structured PlannerReview.
"""
        with model_session(get_planner_model()) as llm:
            structured = llm.with_structured_output(PlannerReview)
            return structured.invoke(prompt)

from src.orchestration.ledger import (
    extract_file_paths_from_goal,
    file_architectural_tier,
    slice_tier1_spec_for_files,
)
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
3. Keep each stage's `objective` to a concise 1-2 sentence summary of that stage's role (exact Tier-1 file specifications are attached automatically).
4. Include EVERY file requested in the Project goal across the stages without inventing unrequested files.

Return only the structured MultiStageBlueprint.
"""
        with model_session(get_planner_model()) as llm:
            structured = llm.with_structured_output(MultiStageBlueprint)
            blueprint = structured.invoke(prompt)

        discovered_files = extract_file_paths_from_goal(project_goal)

        if not blueprint.stages:
            tier1_all = slice_tier1_spec_for_files(
                project_goal, discovered_files, discovered_files
            )
            return MultiStageBlueprint(
                project_summary=project_goal.strip().splitlines()[0],
                stages=[
                    StageGoal(
                        stage_id=1,
                        title="Full Project Implementation",
                        objective=tier1_all,
                        target_files=discovered_files,
                    )
                ],
            )

        cleaned_stages: list[StageGoal] = []
        assigned_files: set[str] = set()
        for idx, stg in enumerate(blueprint.stages[:max_stages], start=1):
            norm_files: list[str] = []
            for f in stg.target_files:
                nf = f.strip().removeprefix("sandbox/").strip("/")
                if nf and nf not in assigned_files:
                    if not discovered_files or nf in discovered_files:
                        norm_files.append(nf)
                        assigned_files.add(nf)
            cleaned_stages.append(
                StageGoal(
                    stage_id=idx,
                    title=stg.title.strip() or f"Stage {idx}",
                    objective=stg.objective.strip(),
                    target_files=norm_files,
                )
            )

        # Ensure every file mentioned in project_goal is assigned to a stage
        for df in discovered_files:
            if df not in assigned_files and cleaned_stages:
                tier = file_architectural_tier(df)
                if tier in (0, 1):
                    target_stg = cleaned_stages[0]
                elif tier == 3:
                    target_stg = cleaned_stages[-1]
                else:
                    mid_idx = min(1, len(cleaned_stages) - 1)
                    target_stg = cleaned_stages[mid_idx]
                target_stg.target_files.append(df)
                assigned_files.add(df)

        cleaned_stages = [s for s in cleaned_stages if s.target_files] or cleaned_stages
        all_stage_files = [f for s in cleaned_stages for f in s.target_files]

        for idx, stg in enumerate(cleaned_stages, start=1):
            stg.stage_id = idx
            tier1_spec = slice_tier1_spec_for_files(
                project_goal, stg.target_files, all_stage_files
            )
            summary_line = stg.objective.splitlines()[0].strip() if stg.objective else stg.title
            if tier1_spec and tier1_spec not in stg.objective:
                stg.objective = (
                    f"{summary_line}\n\n"
                    f"Exact Specification for Stage {stg.stage_id} Files:\n{tier1_spec}"
                )
            elif not stg.objective:
                stg.objective = tier1_spec or project_goal

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

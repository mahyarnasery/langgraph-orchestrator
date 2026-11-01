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
        (e.g. Stage 1: Config/data & core domain models -> Stage 2: Service layer & entry point).
        """
        prompt = f"""
You are the Lead Software Architect decomposing a project into at most {max_stages} sequential build stages.

Project goal:
{project_goal}

Architecture context:
{architecture_context}

Decomposition Rules (CRITICAL):
1. Order stages from foundational config/data files (`.json`, `.csv`, `.toml`) and core domain models/classes up to higher-level services and entry-point scripts (`main.py` / `app.py`).
2. Populate `target_files` for each stage with the exact relative file paths belonging to that stage (e.g. `["config/dam_profile.json", "data/telemetry.csv", "domain/models.py", "domain/controller.py"]`).
3. In each stage's `objective`, preserve the EXACT file names, JSON keys/values, CSV headers/rows, class names, method/function signatures, and formulas from the Project goal for that stage's files! NEVER replace exact specifications with a vague summary.
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
You are the Lead Software Architect / Planner.

Your job is to design the complete architectural blueprint (`Plan`) for the active stage/goal so the Foreman can deterministically dispatch atomic tasks without guessing file paths, imports, or function/method signatures.

Architectural Responsibilities (CRITICAL):
1. Analyze the active stage/project goal and existing workspace repository map.
2. Populate `affected_areas` with ONLY the relative file paths to be created or modified in this stage.
3. Populate `file_blueprints` with one `FileBlueprint` for each target file in this stage:
   - Set `path` to the exact relative file path (e.g. `"config/dam_profile.json"`, `"domain/models.py"`, `"domain/controller.py"`).
   - For non-Python config/data assets (`.json`, `.csv`, `.toml`, etc.): copy the EXACT required JSON keys/values or CSV header/rows into `role`, and leave `required_imports` and `symbols` empty (`[]`).
   - For Python library/model/service modules (`.py`):
     * Keep `role` to a brief 1-sentence summary (NEVER write full Python class/function source code inside `role`!).
     * Populate `required_imports` with the exact import lines needed by the file.
     * Populate `symbols` with one `SymbolSpec` for EVERY standalone function or class method in that file:
       - If the file defines standalone functions, set `target_symbol` to the function name (e.g. `"load_dam_profile"`).
       - If the file defines a class with 2 or more methods, list EACH method as a separate `SymbolSpec` with `target_symbol` set to `"ClassName.method_name"` (e.g. `"TelemetryRecord.__init__"`, `"TelemetryRecord.is_valid"`, `"TelemetryRecord.net_delta"`), starting with `ClassName.__init__`.
       - Preserve exact parameter names, type hints, formulas, and return strings from the Project Specification in `signature` and `contract`.
   - For simple entry-point scripts (`main.py` / `app.py`) that only import, assert, and print: describe the exact steps in `role` and `required_imports`, and leave `symbols` empty (`[]`).
4. Define clear `acceptance_criteria` preserving the exact JSON keys, CSV headers, class/method names, and expected outputs from the Project Specification. NEVER invent substitute keys, columns, or method names.

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

        if validation_result is None:
            validation_text = "Not available"
        elif validation_result.passed:
            stdout_checks = [c for c in validation_result.checks if c.startswith("stdout (")]
            validation_text = (
                f"PASSED: All {len(validation_result.checks)} deterministic checks passed "
                f"(0 errors). Files checked: {', '.join(validation_result.files_checked)}. "
                + (" | ".join(stdout_checks) if stdout_checks else "")
            )
        else:
            validation_text = validation_result.model_dump_json(indent=2)

        prompt = f"""
You are the Planner / Architect conducting the final acceptance review AFTER the Worker and Foreman have finished implementing and assembling the repository.

Project goal:
{project_goal}

Initial architectural objective: {plan.objective}
Acceptance criteria:
{chr(10).join(f'- {c}' for c in plan.acceptance_criteria)}

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

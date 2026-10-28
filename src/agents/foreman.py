from src.orchestration.models import FOREMAN_MODEL, model_session
from src.orchestration.schemas import (
    AssembledModule,
    CycleResult,
    ForemanReview,
    Plan,
    TaskBatch,
    TaskResult,
    ValidationResult,
    WorkerProfile,
)


class ForemanAgent:
    """Implementation coordination, assembly, review, and integration agent."""

    def create_batch(
        self,
        *,
        project_goal: str,
        architecture_context: str,
        worker_profile: WorkerProfile,
        previous_cycle: CycleResult | None,
        plan: Plan,
        ledger_context: str = "",
    ) -> TaskBatch:
        worker_profile_text = worker_profile.model_dump_json(indent=2)

        prompt = f"""
You are the Foreman / Integrator.

Translate the Planner's architectural plan into one executable TaskBatch.

Project goal:
{project_goal}

Architecture context:
{architecture_context}

Execution Ledger:
{ledger_context}

Worker profile:
{worker_profile_text}

Planner plan:
{plan.model_dump_json(indent=2)}

Task Atomization Rules (CRITICAL):
1. Pre-Creation: The orchestration layer automatically pre-creates empty files. NEVER create a task just to create an empty file.
2. Function-Level Atomization for Multi-Function Files:
   - If a Python file requires 2 or more standalone functions (for example, a module with 3 or 4 helper/processing functions), SPLIT that file into separate atomic tasks — ONE task per function!
   - For each function task on that file, set `relevant_files` to that file (e.g. `["analytics.py"]`) and set `target_symbol` to the exact function name (e.g. `"parse_scores"`).
   - If function B calls function A in the same file, list function A's task_id in function B's `dependencies`.
   - The Worker will write each function in an isolated staging file, and you (Foreman) will assemble them into the target module automatically.
3. Single-Task Files & Classes:
   - If a file only has 1 function, or is a simple entry-point script (`main.py` / `app.py`), or defines a single `class` with its methods, keep that file or class as a SINGLE task.
4. Strict File Scope:
   - Each task's `relevant_files` must list ONLY the file that specific task contributes to. Do NOT list imported dependency files in `task.relevant_files`.
5. Preserve exact formulas, f-strings, and return expressions from the Project goal in each task's `acceptance_criteria` (do not paraphrase `len(passing)` or `sum(passing)`).
6. Populate `execution_order` with all `task_id`s in dependency order, and leave `verification_command` as null unless running a dedicated pytest suite without shell pipes.

Every file mentioned by a task must appear in both `task.relevant_files` and `TaskBatch.relevant_files`.


Return only the structured TaskBatch.
"""

        with model_session(FOREMAN_MODEL) as llm:
            structured = llm.with_structured_output(TaskBatch)
            return structured.invoke(prompt)

    def assemble_module(
        self,
        *,
        target_file: str,
        project_goal: str,
        plan: Plan,
        staged_snippets: dict[str, str],
        ast_merged_draft: str,
        ledger_context: str = "",
    ) -> AssembledModule:
        """
        Assemble multiple atomic Worker function snippets into a coherent Python module.
        """
        snippets_text = "\n\n".join(
            f"--- SNIPPET FROM TASK {task_id} ---\n~~~python\n{code}\n~~~"
            for task_id, code in staged_snippets.items()
        )

        prompt = f"""
You are the Foreman / Integrator assembling atomic Worker code snippets into `{target_file}`.

Project goal:
{project_goal}

Planner blueprint:
{plan.model_dump_json(indent=2)}

Execution Ledger:
{ledger_context}

Worker staged snippets for `{target_file}`:
{snippets_text}

Deterministic AST-merged draft of `{target_file}`:
~~~python
{ast_merged_draft}
~~~

Responsibilities:
- Combine all Worker snippets for `{target_file}` into a single, clean, coherent Python file.
- Deduplicate imports at the top of the file.
- Ensure there are NO self-imports (never import `{target_file}` inside `{target_file}`).
- Ensure function calls between snippets in `{target_file}` use matching function names and arguments.
- Preserve every required function and class without adding unrelated test code or extra prints.


Return only the structured AssembledModule.
"""

        with model_session(FOREMAN_MODEL) as llm:
            structured = llm.with_structured_output(AssembledModule)
            return structured.invoke(prompt)

    def create_corrective_batch(
        self,
        *,
        batch: TaskBatch,
        review: ForemanReview,
        task_results: list[TaskResult],
        ledger_context: str = "",
    ) -> TaskBatch:
        """Create a targeted corrective TaskBatch to rebuild/fix failed tasks."""

        prompt = f"""
You are the Foreman creating a corrective TaskBatch for the Worker.

Original batch:
{batch.model_dump_json(indent=2)}

Foreman review and rebuild instructions:
{review.model_dump_json(indent=2)}

Execution Ledger:
{ledger_context}

Previous task execution outcomes:
{', '.join(f'{r.task_id}: {r.status}' for r in task_results)}

Generate a new, focused TaskBatch addressing ONLY the specific tasks needing rebuild/correction.
- Preserve `target_symbol` on function-level tasks so only the broken function is rewritten and re-assembled.
- Populate `execution_order` with the exact `task_id`s in this corrective batch.
- Leave `verification_command` as null.
- Ensure every referenced file is listed in both `task.relevant_files` and `TaskBatch.relevant_files`.


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
        ledger_context: str = "",
    ) -> ForemanReview:
        """Review the assembled implementation and choose the next action."""

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

Project goal:
{project_goal}

Architecture context:
{architecture_context}

Execution Ledger:
{ledger_context}

Planner plan:
{plan.model_dump_json(indent=2)}

Current TaskBatch:
{batch.model_dump_json(indent=2)}

Worker results:
{task_results_context}

Deterministic validation:
{validation_context}

Actual assembled repository contents:
{repository_context}

Worker attempt counts:
{worker_attempts_context}

Foreman direct self-repair attempts used: {foreman_repair_attempts} / {max_foreman_repairs}
{"CRITICAL: You have exhausted all direct self-repair attempts. Do NOT choose 'repair'. Choose 'accept' (if the code works) or 'rebuild'." if foreman_repair_attempts >= max_foreman_repairs else ""}

Determine the smallest appropriate next action:
- "accept": Implementation satisfies the Planner's acceptance criteria and passes deterministic validation.
- "repair": Implementation is close and has a localized issue you (Foreman) can fix directly via `direct_repairs` (prefer `action="overwrite"` with full file content).
- "rebuild": One or more specific Worker tasks need to be sent back to the Worker.


Return only the structured ForemanReview.
"""

        with model_session(FOREMAN_MODEL) as llm:
            structured = llm.with_structured_output(ForemanReview)
            return structured.invoke(prompt)

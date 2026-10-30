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

Task Atomization, Class Method Staging, Package Structure & Verification Rules (CRITICAL):
1. Pre-Creation & Package `__init__.py` Files:
   - The orchestration layer automatically pre-creates empty files AND automatically creates `__init__.py` in every package subdirectory (e.g. `core/__init__.py`, `models/__init__.py`, `services/__init__.py`).
   - NEVER create a task just to create an empty file or an empty `__init__.py`.
2. Multi-Directory Package Paths:
   - Preserve exact relative directory paths from the Project goal (e.g. `"models/product.py"`, `"services/invoice.py"`, `"main.py"`). NEVER flatten subdirectory paths into the root directory.
   - When importing from a package module, use its full dotted package path (e.g. `from models.product import Product`).
3. Function-Level Atomization for Multi-Function Files:
   - If any Python file requires 2 or more standalone functions, SPLIT that file into separate atomic tasks — ONE task per function!
   - Set `relevant_files` to that exact file path (e.g. `["services/invoice.py"]`) and set `target_symbol` to the exact function name (e.g. `"create_products"`).
4. Class Method-Level Atomization for Multi-Method Classes:
   - If a Python file defines a `class` with 2 or more methods (for example, `__init__`, `deposit`, `describe`), SPLIT that class into separate atomic tasks — ONE task per method!
   - For each method task on that class, set `relevant_files` to the class's file (e.g. `["bank.py"]`) and set `target_symbol` to `"ClassName.method_name"` (e.g. `"Account.__init__"`, `"Account.deposit"`, `"Account.describe"`).
   - Put the `"ClassName.__init__"` task first and list its `task_id` in the `dependencies` of the other method tasks on that class.
   - The Worker will write each method in an isolated staging file, and the orchestration layer + you (Foreman) will merge all methods into a single unified `class ClassName:` definition automatically.
5. Single-Task Entry Scripts:
   - If a file only has 1 standalone function or is a simple entry-point script (`main.py` / `app.py`), keep that file as a SINGLE task.
6. Strict File Scope:
   - Each task's `relevant_files` must list ONLY the file that specific task contributes to. Do NOT list imported dependency files in `task.relevant_files`.
   - Do NOT create tasks for files that already exist and are already implemented in the Existing Workspace Repository Map unless the current goal explicitly asks to modify them.
7. Preserve exact formulas, f-strings, and return expressions from the Project goal in each task's `acceptance_criteria`.
8. Deterministic Unit Assertions (`test_assertions`):
   - For every task that implements a function or class method, populate `task.test_assertions` with 1 to 2 executable Python `assert` statements using CONCRETE literal inputs and expected outputs (e.g. `["acc = Account('Alice', 100); assert acc.owner == 'Alice' and acc.balance == 100"]` or `["acc = Account('Alice', 100); assert acc.deposit(50) == 150"]`).
   - Prefer exact input/output values given in the Project goal so expected values are 100% accurate.
   - NEVER use undefined symbolic variables in `test_assertions` — always instantiate objects or call functions with concrete literal arguments.
   - For pure entry-point scripts (`main.py`, `app.py`, `hello.py`) that only print output to stdout, leave `task.test_assertions` empty (`[]`).
9. Populate `execution_order` with all `task_id`s in dependency order, and leave `verification_command` as null unless running a dedicated pytest suite without shell pipes.

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
        Assemble multiple atomic Worker function or class-method snippets into a coherent Python module.
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
- If the snippets implement methods of a class, ensure ALL methods (`__init__` and every other method in the AST-merged draft) are unified inside a single `class` definition without any `pass` stubs overwriting real methods.
- Deduplicate imports at the top of the file.
- Ensure there are NO self-imports (never import `{target_file}` or its module name inside `{target_file}`).
- Preserve cross-package imports (such as `from models.product import Product`) needed by the module.
- Preserve every required function, class, and method without adding unrelated test code or extra prints.


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
        validation_result: ValidationResult | None = None,
        ledger_context: str = "",
    ) -> TaskBatch:
        """Create a targeted corrective TaskBatch to rebuild/fix failed tasks."""
        validation_summary = (
            validation_result.model_dump_json(indent=2)
            if validation_result is not None
            else "Not provided"
        )

        prompt = f"""
You are the Foreman creating a corrective TaskBatch for the Worker.

Original batch:
{batch.model_dump_json(indent=2)}

Deterministic Validator results (note `errors` and `failed_task_ids`):
{validation_summary}

Foreman review and rebuild instructions:
{review.model_dump_json(indent=2)}

Execution Ledger:
{ledger_context}

Previous task execution outcomes:
{', '.join(f'{r.task_id}: {r.status}' for r in task_results)}

Generate a new, focused TaskBatch addressing ONLY the specific tasks needing rebuild/correction:
- Preserve exact package paths in `relevant_files`, `target_symbol` (including `"ClassName.method_name"` for class method tasks), and concrete `test_assertions` so only the broken function/method is rewritten, re-assembled, and re-verified.
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

Deterministic validation (inspect `passed`, `errors`, and `failed_task_ids`):
{validation_context}

Actual assembled repository contents:
{repository_context}

Worker attempt counts:
{worker_attempts_context}

Foreman direct self-repair attempts used: {foreman_repair_attempts} / {max_foreman_repairs}
{"CRITICAL: You have exhausted all direct self-repair attempts. Do NOT choose 'repair'. Choose 'accept' (if all checks pass) or 'rebuild' (targeting `failed_task_ids`)." if foreman_repair_attempts >= max_foreman_repairs else ""}

Determine the smallest appropriate next action:
- "accept": Implementation satisfies the Planner's acceptance criteria and passes all deterministic validation checks and unit assertions.
- "repair": Implementation is close and has a localized issue you (Foreman) can fix directly via `direct_repairs` (prefer `action="overwrite"` with full file content).
- "rebuild": One or more specific Worker tasks (such as those in `validation_result.failed_task_ids`) need to be sent back to the Worker.


Return only the structured ForemanReview.
"""

        with model_session(FOREMAN_MODEL) as llm:
            structured = llm.with_structured_output(ForemanReview)
            return structured.invoke(prompt)

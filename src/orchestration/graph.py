from langgraph.graph import END, StateGraph
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

from .models import (
    FOREMAN_MODEL,
    PLANNER_MODEL,
    WORKER_MODEL,
    model_session,
)

from .state import WorkflowState
from .tools import (
    FOREMAN_TOOLS,
    WORKER_TOOLS,
    create_file,
    edit_file,
    read_file,
    replace_file,
)

from .schemas import (
    CycleResult,
    ForemanReview,
    Plan,
    Task,
    TaskBatch,
    TaskResult,
    ToolAction,
    ValidationResult,
    WorkerProfile,
)

from .validator import validate_batch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _tool_map(tools):
    return {tool.name: tool for tool in tools}


FOREMAN_TOOL_MAP = _tool_map(FOREMAN_TOOLS)
WORKER_TOOL_MAP = _tool_map(WORKER_TOOLS)




# ---------------------------------------------------------------------------
# Planner
# ---------------------------------------------------------------------------


def planner_node(state: WorkflowState) -> dict:
    previous_cycle = (
        state["previous_cycle"].model_dump_json(indent=2)
        if state["previous_cycle"]
        else "None (Cycle 1)"
    )

    worker_profile = state["worker_profile"].model_dump_json(indent=2)

    prompt = f"""
You are the Planner / Architect.

Responsibilities:
- High-level planning only.
- No code generation.
- No repository modification.

Project goal:
{state["project_goal"]}

Architecture context:
{state["architecture_context"]}

Worker profile:
{worker_profile}

Previous cycle:
{previous_cycle}

Produce a compact repository-aware implementation plan.

Return only the structured Plan.
"""

    with model_session(PLANNER_MODEL) as llm:
        structured = llm.with_structured_output(Plan)
        plan = structured.invoke(prompt)

    print("\n=== PLANNER RESULT ===")
    print(plan.model_dump_json(indent=2))

    return {
        "plan": plan,
        "phase": "planned",
    }


# ---------------------------------------------------------------------------
# Foreman
# ---------------------------------------------------------------------------


def foreman_node(state: WorkflowState) -> dict:
    plan = state["plan"]

    if plan is None:
        raise RuntimeError("Planner output missing.")


    previous_cycle = (
        state["previous_cycle"].model_dump_json(indent=2)
        if state["previous_cycle"]
        else "None"
    )

    worker_profile = state["worker_profile"].model_dump_json(indent=2)

    prompt = f"""
You are the Foreman / Integrator.

Project goal:
{state["project_goal"]}

Architecture context:
{state["architecture_context"]}

Worker profile:
{worker_profile}

Previous cycle:
{previous_cycle}

Planner plan:
{plan.model_dump_json(indent=2)}

Create ONE TaskBatch from the Planner's architectural plan.

Your responsibility is to translate the Planner's high-level design
into concrete implementation work.

The Planner decides:
- what the final product should accomplish
- the high-level architecture
- the major components
- the files/modules that should exist
- the acceptance criteria

You decide:
- exact task decomposition
- task ordering
- implementation details
- file preparation
- dependencies between tasks
- how to group related tasks into one Worker batch

Do NOT redesign the architecture established by the Planner.

Important:
- Worker edits existing files only.
- Worker never creates files.
- If a required file does not exist, YOU must create it before Worker execution.
- Every file mentioned in any task's relevant_files MUST be included in TaskBatch.relevant_files.
- TaskBatch.relevant_files must be the complete union of task-level relevant_files.
- Keep Worker tasks small enough for the Worker context limit.
- Group related tasks when doing so allows the Worker to stay in one model session.

Return only the structured TaskBatch.
"""

    with model_session(FOREMAN_MODEL) as llm:
        structured = llm.with_structured_output(TaskBatch)
        batch = structured.invoke(prompt)

        batch.relevant_files = sorted(
            {
                path
                for task in batch.tasks
                for path in task.relevant_files
            }
        )

        print("\n=== FOREMAN RESULT ===")
        print(batch.model_dump_json(indent=2))

        print("\n=== FOREMAN REPOSITORY PREPARATION ===")

        for relative_path in batch.relevant_files:
            if relative_path in ("", ".", ".."):
                continue

            try:
                read_file.func(path=relative_path)
                print(f"{relative_path} already exists")

            except FileNotFoundError:
                create_file.func(
                    path=relative_path,
                    content="",
                )
                print(f"Created {relative_path}")

    return {
        "task_batches": [batch],
        "phase": "task_ready",
    }


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------


def _execute_task(
    llm_with_tools,
    batch: TaskBatch,
    task: Task,
) -> TaskResult:
    """
    Execute one task inside the already-loaded Worker session.

    Alpha02:
    TaskResult is built from observed tool execution rather than
    Worker claims.
    """

    prompt = f"""
You are the Worker.

Execute ONLY this task.

Batch objective:
{batch.objective}

Shared context:
{batch.shared_context}

Task:
{task.model_dump_json(indent=2)}

Rules:

- Read the target file before editing it.
- Edit existing files only.
- Never create files.
- Never redesign architecture.
- Never modify files outside the task scope.

The edit_file tool has TWO valid modes:

MODE 1 — Initialize an empty file
Use this ONLY if read_file returned an empty string.

Arguments:
- path = target file
- old_text = ""
- new_text = complete file contents

MODE 2 — Replace existing code
Use this when the file already contains code.

Arguments:
- path = target file
- old_text = exact existing substring copied from read_file
- new_text = replacement text

Requirements:
- Never use Markdown fences.
- Never use None or null.
- old_text must exactly match the current file.
- If EDIT_SUCCESS occurs, continue with the next task.
- If OLD_TEXT_NOT_FOUND occurs, read the file again before retrying.
- Do not repeatedly retry the same failed replacement.
"""

    messages = [HumanMessage(content=prompt)]

    tool_actions: list[ToolAction] = []
    files_changed: set[str] = set()
    errors: list[str] = []

    for iteration in range(5):
        print(
            f"\\n=== {task.task_id} | iteration {iteration+1}/5 ==="
        )

        response = llm_with_tools.invoke(messages)

        if not response.tool_calls:

            status = (
                "completed"
                if (
                    not errors
                    and (
                        files_changed
                        or not any(
                            a.tool == "edit_file"
                            for a in tool_actions
                        )
                    )
                )
                else "failure"
            )

            result = TaskResult(
                task_id=task.task_id,
                status=status,
                tool_actions=tool_actions,
                files_changed=sorted(files_changed),
                validation=[],
                errors=errors,
            )

            print("\\n=== TASK RESULT ===")
            print(result.model_dump_json(indent=2))

            return result

        messages.append(response)

        for tool_call in response.tool_calls:

            name = tool_call["name"]

            if name not in WORKER_TOOL_MAP:
                raise RuntimeError(
                    f"Unauthorized tool: {name}"
                )

            try:
                tool_result = WORKER_TOOL_MAP[name].func(
                    **tool_call["args"]
                )

            except Exception as exc:

                error_message = (
                    f"{type(exc).__name__}: {exc}"
                )

                tool_result = {
                    "tool": name,
                    "path": tool_call["args"].get("path"),
                    "result": "TOOL_ERROR",
                    "content": error_message,
                }

                errors.append(error_message)

            action = ToolAction(
                tool=tool_result["tool"],
                path=tool_result.get("path"),
                result=tool_result["result"],
            )

            tool_actions.append(action)

            if tool_result["result"] == "EDIT_SUCCESS":
                files_changed.add(tool_result["path"])

            elif tool_result["result"] in (
                "OLD_TEXT_NOT_FOUND",
                "AMBIGUOUS_EDIT",
            ):
                errors.append(
                    f"{tool_result['result']} on {tool_result['path']}"
                )

            elif tool_result["result"] == "TOOL_ERROR":
                errors.append(
                    f"{name} failed on {tool_result.get('path')}"
                )

            if tool_result["result"] == "TOOL_ERROR":
                tool_message_content = (
                    "TOOL_ERROR: "
                    + tool_result.get(
                        "content",
                        "Unknown tool error.",
                    )
                    + "\n"
                    "Do not repeat the same tool call. "
                    "Inspect the file again and construct a valid "
                    "tool call with all required arguments."
                )

            elif (
                tool_result["tool"] == "read_file"
                and tool_result["result"] == "OK"
            ):
                tool_message_content = tool_result["content"]

            else:
                tool_message_content = tool_result["result"]

            messages.append(
                ToolMessage(
                    content=tool_message_content,
                    tool_call_id=tool_call["id"],
                )
            )

    return TaskResult(
        task_id=task.task_id,
        status="failure",
        tool_actions=tool_actions,
        files_changed=sorted(files_changed),
        validation=[],
        errors=errors + ["iteration_limit_exceeded"],
    )


def worker_node(state: WorkflowState) -> dict:
    """
    Execute ALL tasks inside one TaskBatch using ONE Granite session.
    """

    if not state["task_batches"]:
        raise RuntimeError("No TaskBatch received.")

    batch = state["task_batches"][0]

    results: list[TaskResult] = []

    print("\n=== WORKER SESSION START ===")
    print(f"Batch: {batch.batch_id}")

    with model_session(WORKER_MODEL) as llm:
        llm_with_tools = llm.bind_tools(WORKER_TOOLS)

        ordered = []

        for task_id in batch.execution_order:
            for task in batch.tasks:
                if task.task_id == task_id:
                    ordered.append(task)
                    break

        for task in ordered:
            print(f"\n----- Executing {task.task_id} -----")

            result = _execute_task(
                llm_with_tools,
                batch,
                task,
            )

            results.append(result)

    print("\n=== WORKER SESSION COMPLETE ===")

    attempts = dict(state["worker_attempts"])

    for result in results:
        attempts[result.task_id] = attempts.get(result.task_id, 0) + 1

    return {
    "task_results": results,
    "worker_attempts": attempts,
    "phase": "worker_complete",
}


# ---------------------------------------------------------------------------
# Validator
# ---------------------------------------------------------------------------

def validator_node(state: WorkflowState) -> dict:
    """
    Deterministically validate the completed Worker batch.

    No LLMs are used here. The validator produces objective evidence that
    will later be consumed by the Foreman review.
    """

    if not state["task_batches"]:
        raise RuntimeError("No TaskBatch available for validation.")

    batch = state["task_batches"][0]

    validation = validate_batch(
        batch=batch,
        task_results=state["task_results"],
    )

    print("\n=== VALIDATOR RESULT ===")
    print(validation.model_dump_json(indent=2))

    return {
        "validation_result": validation,
        "phase": "validated",
    }


def foreman_review_node(state: WorkflowState) -> dict:
    """
    Review the current Worker implementation using:
    - observed Worker execution evidence
    - deterministic Validator evidence
    - actual repository contents

    The Foreman does NOT create a corrective TaskBatch here.

    Instead it decides whether the implementation should be:
        accept  -> finished
        repair  -> Foreman directly repairs the repository
        rebuild -> a failed task is cleared and prepared for a fresh Worker attempt
    """

    if not state["task_batches"]:
        raise RuntimeError("No TaskBatch available for Foreman review.")

    if not state["task_results"]:
        raise RuntimeError("No TaskResults available for Foreman review.")

    validation = state["validation_result"]

    if validation is None:
        raise RuntimeError("ValidationResult missing before Foreman review.")

    plan = state["plan"]
    batch = state["task_batches"][0]
    results = state["task_results"]

    if plan is None:
        raise RuntimeError("Planner output missing.")

    review_context = "\n\n".join(
        f"TaskResult {result.task_id}:\n"
        f"{result.model_dump_json(indent=2)}"
        for result in results
    )

    relevant_files = sorted(
        set(batch.relevant_files)
        | {
            action.path
            for result in results
            for action in result.tool_actions
            if action.path
        }
    )

    print("\n=== FOREMAN REVIEW: REPOSITORY INSPECTION ===")

    repository_evidence: list[str] = []

    for relative_path in relevant_files:
        try:
            tool_result = read_file.func(path=relative_path)

            repository_evidence.append(
                f"FILE: {relative_path}\n"
                f"{tool_result['content']}"
            )

            print(f"Inspected: {relative_path}")

        except Exception as exc:
            repository_evidence.append(
                f"FILE: {relative_path}\n"
                f"READ_ERROR: {type(exc).__name__}: {exc}"
            )

            print(
                f"Failed to inspect {relative_path}: "
                f"{type(exc).__name__}: {exc}"
            )

    validation_context = validation.model_dump_json(indent=2)
    repository_context = "\n\n".join(repository_evidence)

    worker_attempts = state["worker_attempts"]

    prompt = f"""
You are the Foreman / Integrator performing an implementation review.

Your responsibility is to inspect the Worker result and determine the
smallest appropriate next action.

The Planner defines architecture.
You are responsible for implementation-level integration and repair.

Project goal:
{state["project_goal"]}

Architecture context:
{state["architecture_context"]}

Planner plan:
{plan.model_dump_json(indent=2)}

Current TaskBatch:
{batch.model_dump_json(indent=2)}

Worker TaskResults:
{review_context}

DETERMINISTIC VALIDATION:
{validation_context}

ACTUAL REPOSITORY CONTENTS:
{repository_context}

WORKER ATTEMPT COUNTS:
{worker_attempts}

IMPORTANT RULES:

1. The deterministic Validator is authoritative for:
   - syntax
   - file existence
   - scope compliance

2. The actual repository contents are the source of truth for implementation.

3. Worker TaskResults are execution evidence.
   Do not assume that "completed" means the implementation is correct.

4. Every reported problem must be concrete.
   Identify the affected file and function/area whenever possible.

5. Do NOT redesign the architecture.
   The Planner owns architecture.

6. Choose exactly ONE decision:

   "accept"
       Use when the implementation satisfies the Planner's acceptance
       criteria and there is no meaningful implementation problem.

   "repair"
       Use when the implementation is fundamentally correct but has a
       localized problem that the Foreman can fix directly and efficiently.
       Examples:
       - small syntax or logic mistake
       - incorrect import
       - missing argument
       - simple integration mistake
       - small mismatch with acceptance criteria

   "rebuild"
       Use when a Worker task should be attempted again from a clean file.
       This is appropriate when:
       - the Worker repeatedly failed to edit the existing implementation
       - the existing implementation is confusing or corrupted
       - the task requires a substantially different implementation
       - continuing to patch the existing file would be less reliable than
         starting clean

7. Prefer "repair" over "rebuild" when the problem is small.

8. Prefer "rebuild" over repeatedly asking the Worker to patch a bad
   implementation.

9. If a task has already failed three Worker attempts, do NOT send that
   task back to the Worker again. Use "repair" if the Foreman can fix it
   directly.

10. For "repair":
    - accepted MUST be false
    - decision MUST be "repair"
    - provide precise repair_instructions
    - rebuild_task_ids MUST be empty

11. For "rebuild":
    - accepted MUST be false
    - decision MUST be "rebuild"
    - provide precise repair_instructions explaining WHY the previous
      implementation failed and exactly what the new implementation must
      avoid
    - rebuild_task_ids MUST contain the IDs of tasks that need rebuilding

12. The rebuild instructions must use the reason for the original failure.
    Do not simply repeat the original task description.

13. For "accept":
    - accepted MUST be true
    - decision MUST be "accept"
    - problems MUST be empty
    - repair_instructions MUST be empty
    - rebuild_task_ids MUST be empty

14. Do NOT create a TaskBatch in this review.
    The orchestration layer will handle repair/rebuild according to your
    decision.

Return only the structured ForemanReview.
"""

    with model_session(FOREMAN_MODEL) as llm:
        structured = llm.with_structured_output(ForemanReview)
        review = structured.invoke(prompt)

    if review.decision not in {
        "accept",
        "repair",
        "rebuild",
    }:
        raise RuntimeError(
            f"Invalid Foreman review decision: {review.decision}"
        )

    if review.decision == "accept":
        review.accepted = True
        review.problems = []
        review.repair_instructions = []
        review.rebuild_task_ids = []

    else:
        review.accepted = False

        if review.decision == "repair":
            review.rebuild_task_ids = []

            # --------------------------------------------------
            # Fallback: Granite sometimes returns decision="repair"
            # but leaves repair_instructions empty.
            # Convert validator errors into concrete instructions.
            # --------------------------------------------------
            if not review.repair_instructions:
                review.repair_instructions = []

                for error in validation.errors:
                    if "main.py" in error:
                        review.repair_instructions.append(
                            "Repair main.py and replace any smart quotes (U+201C/U+201D) with normal ASCII double quotes."
                        )

                    elif "scraper.py" in error:
                        review.repair_instructions.append(
                            "Repair scraper.py and replace any smart quotes (U+201C/U+201D) with normal ASCII double quotes."
                        )

                    elif "processor.py" in error:
                        review.repair_instructions.append(
                            "Repair processor.py by correcting the malformed function structure and restoring valid Python syntax."
                        )

                if not review.repair_instructions:
                    review.repair_instructions.append(
                        "Repair the files identified by the validator and eliminate all reported syntax errors."
                    )

        elif review.decision == "rebuild":
            if not review.rebuild_task_ids:
                raise RuntimeError(
                    "Foreman selected rebuild without specifying "
                    "rebuild_task_ids."
                )

    print("\n=== FOREMAN REVIEW ===")
    print(review.model_dump_json(indent=2))

    return {
        "foreman_review": review,
        "phase": (
            "foreman_review_accepted"
            if review.accepted
            else f"foreman_review_{review.decision}"
        ),
    }

def foreman_rebuild_node(state: WorkflowState) -> dict:
    """
    Prepare failed Worker tasks for a clean rebuild.

    The Foreman:
    1. identifies the failed tasks selected by the review
    2. clears their target files
    3. creates a new TaskBatch containing only those tasks
    4. injects the review's failure analysis into the task context

    The Worker then receives the rebuilt task from a clean file rather
    than trying to patch the previous failed implementation.
    """

    review = state["foreman_review"]

    if review is None:
        raise RuntimeError("Foreman review missing before rebuild.")

    if review.decision != "rebuild":
        raise RuntimeError(
            "foreman_rebuild_node requires a 'rebuild' Foreman decision."
        )

    if not state["task_batches"]:
        raise RuntimeError("No TaskBatch available for rebuild.")

    original_batch = state["task_batches"][0]

    rebuild_ids = set(review.rebuild_task_ids)

    if not rebuild_ids:
        raise RuntimeError(
            "No task IDs were provided for rebuild."
        )

    original_tasks = {
        task.task_id: task
        for task in original_batch.tasks
    }

    missing_ids = rebuild_ids - original_tasks.keys()

    if missing_ids:
        raise RuntimeError(
            "Foreman requested rebuild for unknown task IDs: "
            + ", ".join(sorted(missing_ids))
        )

    selected_tasks = [
        original_tasks[task_id]
        for task_id in original_batch.execution_order
        if task_id in rebuild_ids
    ]

    if not selected_tasks:
        raise RuntimeError(
            "No valid tasks selected for rebuild."
        )

    print("\n=== FOREMAN REBUILD PREPARATION ===")

    # ------------------------------------------------------------
    # Determine which files belong to the rebuilt tasks.
    # ------------------------------------------------------------

    rebuild_files = sorted(
        {
            path
            for task in selected_tasks
            for path in task.relevant_files
        }
    )

    # ------------------------------------------------------------
    # Clear existing implementations.
    #
    # The Worker cannot create files, so the Foreman must ensure
    # the target file exists and is empty before the Worker starts.
    # ------------------------------------------------------------

    for relative_path in rebuild_files:
        if relative_path in ("", ".", ".."):
            continue

        try:
            current = read_file.func(path=relative_path)

            current_content = current["content"]

            if current_content.strip():
                result = edit_file.func(
                    path=relative_path,
                    old_text=current_content,
                    new_text="",
                )

                if result["result"] != "EDIT_SUCCESS":
                    raise RuntimeError(
                        f"Failed to clear {relative_path}: "
                        f"{result['result']}"
                    )

                print(f"Cleared {relative_path}")

            else:
                print(f"{relative_path} is already empty")

        except FileNotFoundError:
            create_file.func(
                path=relative_path,
                content="",
            )

            print(f"Created empty rebuild file: {relative_path}")

    # ------------------------------------------------------------
    # Build replacement tasks.
    #
    # Do not blindly reuse the old objectives. Add the Foreman's
    # failure analysis so the Worker knows exactly what went wrong.
    # ------------------------------------------------------------

    rebuild_context = list(review.repair_instructions)

    rebuilt_tasks: list[Task] = []

    for task in selected_tasks:
        rebuilt_tasks.append(
            task.model_copy(
                update={
                    "objective": (
                        f"Rebuild task {task.task_id} from a clean file. "
                        f"Do not preserve the previous failed implementation."
                    ),
                    "constraints": (
                        list(task.constraints)
                        + [
                            "The target file has been cleared by the Foreman.",
                            "Implement the task from a clean slate.",
                            "Do not attempt to recover or patch the previous implementation.",
                            *rebuild_context,
                        ]
                    ),
                }
            )
        )

    rebuilt_batch = TaskBatch(
        batch_id=f"{original_batch.batch_id}_rebuild",
        objective=(
            f"Rebuild failed tasks from TaskBatch "
            f"{original_batch.batch_id} using the Foreman's failure analysis."
        ),
        tasks=rebuilt_tasks,
        shared_context=(
            list(original_batch.shared_context)
            + [
                "This is a clean rebuild after Worker failure.",
                *review.repair_instructions,
            ]
        ),
        relevant_files=rebuild_files,
        constraints=list(original_batch.constraints),
        execution_order=[
            task.task_id
            for task in rebuilt_tasks
        ],
    )

    print("\n=== FOREMAN REBUILT BATCH ===")
    print(rebuilt_batch.model_dump_json(indent=2))

    return {
        "task_batches": [rebuilt_batch],
        "task_results": [],
        "phase": "task_ready",
    }


def foreman_repair_node(state: WorkflowState) -> dict:
    """
    Allow the Foreman to directly repair a small implementation problem.

    The Foreman operates through the normal LangChain tool-calling protocol:

        AIMessage
            -> tool execution
            -> ToolMessage(tool_call_id=...)
            -> next AIMessage

    Python owns the repair state machine and termination conditions.

    A repair is considered successful only when:

        1. An edit/create operation succeeds.
        2. The affected file is reread successfully.
        3. The verification read confirms that the repository is readable
           after the edit.

    The LLM is never given another iteration after a successful edit has
    been deterministically verified. This prevents the model from
    repeating an already-successful repair.

    Failed edits are returned to the LLM through normal ToolMessages so
    that the model can reread the file and construct a corrected edit.

    The exact same failed edit is never executed twice.
    """

    review = state["foreman_review"]

    if review is None:
        raise RuntimeError("Foreman review missing before repair.")

    if review.decision != "repair":
        raise RuntimeError(
            "foreman_repair_node requires a 'repair' Foreman decision."
        )

    if not review.repair_instructions:
        raise RuntimeError(
            "Foreman selected repair but provided no repair instructions."
        )

    if not state["task_batches"]:
        raise RuntimeError("No TaskBatch available for Foreman repair.")

    # ------------------------------------------------------------------
    # Repair-attempt accounting
    # ------------------------------------------------------------------

    repair_attempts = state["foreman_repair_attempts"] + 1

    if repair_attempts > 3:
        raise RuntimeError(
            "Foreman repair attempts exhausted after 3 attempts."
        )

    batch = state["task_batches"][0]

    # ------------------------------------------------------------------
    # Determine which files the Foreman should inspect.
    #
    # Include both:
    #
    #   1. files declared relevant by the TaskBatch
    #   2. files actually touched by Workers
    #
    # This prevents the Foreman from being blind to a file that was
    # modified even if TaskBatch metadata was incomplete.
    # ------------------------------------------------------------------

    relevant_files = sorted(
        set(batch.relevant_files)
        | {
            action.path
            for result in state["task_results"]
            for action in result.tool_actions
            if action.path
        }
    )

    if not relevant_files:
        raise RuntimeError(
            "Foreman repair has no repository files available for inspection."
        )

    print("\n=== FOREMAN DIRECT REPAIR ===")
    print(f"Repair attempt: {repair_attempts}/3")

    # ------------------------------------------------------------------
    # Initial repository inspection.
    #
    # This is deliberately performed by Python before the LLM session,
    # exactly as the previous implementation did.
    # ------------------------------------------------------------------

    repository_evidence: list[str] = []

    for relative_path in relevant_files:
        try:
            result = read_file.invoke({"path": relative_path})

            if (
                isinstance(result, dict)
                and result.get("result") == "OK"
            ):
                repository_evidence.append(
                    f"FILE: {relative_path}\n"
                    f"{result.get('content', '')}"
                )

                print(f"Inspected: {relative_path}")

            else:
                repository_evidence.append(
                    f"FILE: {relative_path}\n"
                    f"READ_ERROR: "
                    f"{result.get('error', result)}"
                )

        except Exception as exc:
            repository_evidence.append(
                f"FILE: {relative_path}\n"
                f"READ_ERROR: {type(exc).__name__}: {exc}"
            )

    repository_context = "\n\n".join(repository_evidence)

    # ------------------------------------------------------------------
    # Repair prompt.
    # ------------------------------------------------------------------

    prompt = f"""
        You are the Foreman / Integrator performing a direct repository repair.

        The implementation has already been reviewed.

        The Foreman Review determined that the problem is small enough to repair
        directly rather than sending the task back to the Worker.

        Project goal:
        {state["project_goal"]}

        Architecture context:
        {state["architecture_context"]}

        Current TaskBatch:
        {batch.model_dump_json(indent=2)}

        Foreman Review:
        {review.model_dump_json(indent=2)}

        Current repository contents:
        {repository_context}

        Repair instructions:
        {chr(10).join(f"- {instruction}" for instruction in review.repair_instructions)}

        This is direct repair attempt {repair_attempts} of 3.

        Your job is to actually repair the repository using the available tools.

        IMPORTANT RULES:

        1. Make ONLY the changes necessary to satisfy the repair instructions.

        2. Do NOT redesign the architecture.

        3. Do NOT create a new TaskBatch.

        4. Do NOT send the task back to the Worker.

        5. Preserve correct existing implementation.

        6. Before editing a file, READ the current file with read_file.

        7. Never guess old_text for edit_file.

        8. For edit_file, old_text MUST exactly match text currently present
        in the file.

        9. If edit_file returns OLD_TEXT_NOT_FOUND:
        - do NOT repeat the same edit;
        - read the file again;
        - inspect the newly returned contents;
        - construct a new exact edit from those contents.

        10. If edit_file returns AMBIGUOUS_EDIT:
            - do NOT repeat the same edit;
            - read the file again;
            - construct a more specific exact old_text.

        11. If a tool returns TOOL_ERROR:
            - do NOT blindly repeat the same tool call;
            - inspect the relevant file again;
            - recover using the actual current repository contents.

        12. If an edit succeeds, do NOT perform another edit on the same repair.
            The repository will be verified automatically after the successful edit.

        13. Do not merely explain what should be changed.
            Perform the change with the tools.

        14. If the requested repair is already present, read the relevant file and
            report the current state. Do not manufacture an unnecessary edit.

        15. Keep the repair localized to the files identified by the review.

        16. Never repeat an identical failed edit.

        17. Once the requested repair has been performed, stop calling tools.
        """

    # ------------------------------------------------------------------
    # Repair session state.
    # ------------------------------------------------------------------

    max_iterations = 5

    # True only after a successful repository modification has been
    # followed by a successful deterministic read.
    repair_verified = False

    # Exact edit calls that have already failed.
    failed_edit_calls: set[tuple[str, str, str]] = set()

    # Exact edit calls that have already succeeded.
    #
    # Normally the session terminates immediately after verification,
    # but this also protects against multiple edit calls appearing in
    # the same AI response.
    successful_edit_calls: set[tuple[str, str, str]] = set()

    # ------------------------------------------------------------------
    # Tool execution helper.
    #
    # This converts every repository-tool result into a ToolMessage
    # compatible with the exact AIMessage -> ToolMessage protocol already
    # used by _execute_task().
    # ------------------------------------------------------------------

    def _tool_message_content(
        tool_name: str,
        tool_result: object,
    ) -> str:
        if not isinstance(tool_result, dict):
            return str(tool_result)

        result_status = tool_result.get("result")

        if result_status == "TOOL_ERROR":
            return (
                "TOOL_ERROR: "
                + str(
                    tool_result.get(
                        "error",
                        tool_result.get(
                            "content",
                            "Unknown tool error.",
                        ),
                    )
                )
                + "\n\n"
                "Do not repeat the same tool call. "
                "Inspect the current repository state and recover "
                "using the actual file contents."
            )

        if result_status == "OLD_TEXT_NOT_FOUND":
            return (
                "OLD_TEXT_NOT_FOUND: "
                "The supplied old_text does not exactly match the "
                "current file contents.\n\n"
                "Do NOT repeat the same edit. "
                "Read the file again and construct old_text from "
                "the newly returned contents."
            )

        if result_status == "AMBIGUOUS_EDIT":
            return (
                "AMBIGUOUS_EDIT: "
                "The supplied old_text matched more than once in "
                "the current file.\n\n"
                "Do NOT repeat the same edit. "
                "Read the file again and use a more specific exact "
                "old_text."
            )

        elif result_status == "NO_CHANGE":
            tool_message_content = (
                "NO_CHANGE: The requested replacement matched, but it produced "
                "no modification to the repository.\n\n"
                "The file still contains the problem.\n"
                "Read the file again and choose a different exact old_text/new_text "
                "that actually changes the file."
            )

        if result_status == "DUPLICATE_EDIT":
            return (
                "DUPLICATE_EDIT: "
                "This exact edit has already been executed successfully. "
                "Do not repeat it. Read the file and verify the current "
                "repository state instead."
            )

        if (
            tool_name == "read_file"
            and result_status == "OK"
        ):
            return str(tool_result.get("content", ""))

        if result_status == "EDIT_SUCCESS":
            return (
                "EDIT_SUCCESS: "
                "The requested edit was successfully applied."
            )

        if result_status in {
            "CREATE_SUCCESS",
            "OK",
        }:
            return str(
                tool_result.get(
                    "content",
                    result_status,
                )
            )

        return str(tool_result)

    # ------------------------------------------------------------------
    # Tool execution.
    #
    # IMPORTANT:
    #
    # We intentionally use FOREMAN_TOOL_MAP, exactly like the Worker
    # execution path, instead of directly hard-coding each tool call.
    # ------------------------------------------------------------------

    with model_session(FOREMAN_MODEL) as llm:
        repair_llm = llm.bind_tools(FOREMAN_TOOLS)

        messages = [
            SystemMessage(
                content=(
                    "You are the Foreman. "
                    "You are responsible for directly repairing the "
                    "repository. "
                    "Use the provided repository tools instead of "
                    "merely describing changes. "
                    "Always inspect current file contents before editing. "
                    "Never repeat a failed edit."
                )
            ),
            HumanMessage(content=prompt),
        ]

        for iteration in range(max_iterations):
            print(
                f"\n=== FOREMAN REPAIR | iteration "
                f"{iteration + 1}/{max_iterations} ==="
            )

            # ----------------------------------------------------------
            # Ask Granite for the next repository action.
            # ----------------------------------------------------------

            response = repair_llm.invoke(messages)

            # ----------------------------------------------------------
            # Preserve the exact AIMessage in the conversation.
            # ----------------------------------------------------------

            messages.append(response)

            # ----------------------------------------------------------
            # No tool calls.
            #
            # If the model stops without having performed a verified
            # edit, the repair is NOT successful.
            #
            # This prevents a textual "done" response from being treated
            # as repository evidence.
            # ----------------------------------------------------------

            if not response.tool_calls:
                print(
                    "*** FOREMAN REPAIR: "
                    "LLM produced no tool calls."
                )

                if repair_verified:
                    break

                print(
                    "*** FOREMAN REPAIR: "
                    "No verified repository repair exists."
                )
                break

            # ----------------------------------------------------------
            # Execute every requested tool call.
            #
            # Every executed call gets exactly one ToolMessage carrying
            # the corresponding tool_call_id.
            # ----------------------------------------------------------

            successful_edit_path: str | None = None

            for tool_call in response.tool_calls:
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]

                print(
                    f"\n*** FOREMAN REPAIR TOOL: "
                    f"{tool_name}({tool_args}) ***"
                )

                # ------------------------------------------------------
                # Authorization.
                # ------------------------------------------------------

                if tool_name not in FOREMAN_TOOL_MAP:
                    tool_result = {
                        "tool": tool_name,
                        "path": tool_args.get("path"),
                        "result": "TOOL_ERROR",
                        "content": (
                            "Unauthorized Foreman repair tool: "
                            f"{tool_name}"
                        ),
                    }

                # ------------------------------------------------------
                # EDIT FILE.
                # ------------------------------------------------------

                elif tool_name == "edit_file":
                    path = str(tool_args.get("path", ""))
                    old_text = str(tool_args.get("old_text", ""))
                    new_text = str(tool_args.get("new_text", ""))

                    edit_signature = (
                        path,
                        old_text,
                        new_text,
                    )

                    # ----------------------------------------------
                    # Prevent repeating an edit that already succeeded.
                    # ----------------------------------------------

                    if edit_signature in successful_edit_calls:
                        tool_result = {
                            "tool": "edit_file",
                            "path": path,
                            "result": "DUPLICATE_EDIT",
                            "content": (
                                "This exact edit already succeeded."
                            ),
                        }

                    # ----------------------------------------------
                    # Prevent repeating an edit that already failed.
                    # ----------------------------------------------

                    elif edit_signature in failed_edit_calls:
                        tool_result = {
                            "tool": "edit_file",
                            "path": path,
                            "result": "TOOL_ERROR",
                            "content": (
                                "This exact edit already failed during "
                                "the current repair. "
                                "Read the file again before attempting "
                                "a different edit."
                            ),
                        }

                    else:
                        try:
                            tool_result = FOREMAN_TOOL_MAP[
                                tool_name
                            ].func(
                                **tool_args
                            )

                        except Exception as exc:
                            tool_result = {
                                "tool": tool_name,
                                "path": path,
                                "result": "TOOL_ERROR",
                                "content": (
                                    f"{type(exc).__name__}: {exc}"
                                ),
                            }

                    # ----------------------------------------------
                    # Record edit outcome.
                    # ----------------------------------------------

                    if isinstance(tool_result, dict):
                        result_status = tool_result.get("result")
                    else:
                        result_status = None

                    if result_status == "EDIT_SUCCESS":
                        successful_edit_calls.add(edit_signature)

                        successful_edit_path = path

                    elif result_status in {
                        "OLD_TEXT_NOT_FOUND",
                        "AMBIGUOUS_EDIT",
                        "NO_CHANGE",
                        "TOOL_ERROR",
                    }:
                        failed_edit_calls.add(edit_signature)

                # ------------------------------------------------------
                # CREATE FILE.
                # ------------------------------------------------------

                elif tool_name == "create_file":
                    path = str(tool_args.get("path", ""))

                    try:
                        tool_result = FOREMAN_TOOL_MAP[
                            tool_name
                        ].func(
                            **tool_args
                        )

                    except Exception as exc:
                        tool_result = {
                            "tool": tool_name,
                            "path": path,
                            "result": "TOOL_ERROR",
                            "content": (
                                f"{type(exc).__name__}: {exc}"
                            ),
                        }

                    if isinstance(tool_result, dict):
                        if tool_result.get("result") in {
                            "CREATE_SUCCESS",
                            "EDIT_SUCCESS",
                        }:
                            successful_edit_path = path

                # ------------------------------------------------------
                # REPLACE FILE.
                # ------------------------------------------------------

                elif tool_name == "replace_file":
                    path = str(tool_args.get("path", ""))
                    content = str(tool_args.get("content", ""))

                    try:
                        tool_result = FOREMAN_TOOL_MAP[
                            tool_name
                        ].func(
                            **tool_args
                        )

                    except Exception as exc:
                        tool_result = {
                            "tool": "replace_file",
                            "path": path,
                            "result": "TOOL_ERROR",
                            "content": (
                                f"{type(exc).__name__}: {exc}"
                            ),
                        }

                    if isinstance(tool_result, dict):
                        result_status = tool_result.get("result")
                    else:
                        result_status = None

                    if result_status == "EDIT_SUCCESS":
                        successful_edit_path = path

            
                # ------------------------------------------------------
                # READ FILE.
                # ------------------------------------------------------

                else:
                    try:
                        tool_result = FOREMAN_TOOL_MAP[
                            tool_name
                        ].func(
                            **tool_args
                        )

                    except Exception as exc:
                        tool_result = {
                            "tool": tool_name,
                            "path": tool_args.get("path"),
                            "result": "TOOL_ERROR",
                            "content": (
                                f"{type(exc).__name__}: {exc}"
                            ),
                        }

                # ------------------------------------------------------
                # Print tool result.
                # ------------------------------------------------------

                print(
                    f"*** TOOL RESULT: "
                    f"{tool_result} ***"
                )

                # ------------------------------------------------------
                # Append EXACTLY ONE ToolMessage for this tool call.
                #
                # This is the same protocol already used by the Worker.
                # ------------------------------------------------------

                messages.append(
                    ToolMessage(
                        content=_tool_message_content(
                            tool_name,
                            tool_result,
                        ),
                        tool_call_id=tool_call["id"],
                    )
                )

                # ------------------------------------------------------
                # If an edit succeeded, we do not execute additional
                # model-requested modifications from the same response.
                #
                # The successful edit is enough to trigger deterministic
                # verification immediately below.
                #
                # Any remaining tool calls in the AI response are ignored
                # because continuing to modify the repository after a
                # successful repair would violate the "small localized
                # repair" contract.
                # ------------------------------------------------------

                if successful_edit_path is not None:
                    break

            # ----------------------------------------------------------
            # Deterministic verification after successful modification.
            #
            # This read is controlled by Python, not requested by Granite.
            #
            # IMPORTANT:
            #
            # It is NOT inserted into messages because there is no
            # corresponding AI tool_call for it. The LLM does not need
            # another turn after successful verification.
            # ----------------------------------------------------------

            if successful_edit_path is not None:
                path = successful_edit_path

                print(
                    f"*** FOREMAN REPAIR VERIFY: "
                    f"read_file({path}) ***"
                )

                try:
                    verification_result = read_file.invoke(
                        {"path": path}
                    )

                except Exception as exc:
                    verification_result = {
                        "result": "TOOL_ERROR",
                        "error": (
                            f"{type(exc).__name__}: {exc}"
                        ),
                    }

                print(
                    f"*** FOREMAN REPAIR VERIFY RESULT: "
                    f"{verification_result} ***"
                )

                if (
                    isinstance(verification_result, dict)
                    and verification_result.get("result") == "OK"
                ):
                    repair_verified = True

                    print(
                        "*** FOREMAN REPAIR: "
                        "Successful edit verified."
                    )

                    # --------------------------------------------------
                    # CRITICAL:
                    #
                    # Stop the LLM session immediately.
                    #
                    # This prevents the exact Cycle 2 failure:
                    #
                    #   edit
                    #   -> read
                    #   -> edit same thing again
                    #
                    # The model never receives another invocation after
                    # a successful verified repair.
                    # --------------------------------------------------

                    break

                print(
                    "*** FOREMAN REPAIR: "
                    "Edit succeeded but verification failed."
                )

                # ------------------------------------------------------
                # Verification failure is NOT success.
                #
                # The LLM may receive another turn and inspect the file.
                # The original successful edit ToolMessage is already in
                # the conversation, so the tool-call protocol remains
                # valid.
                # ------------------------------------------------------

    # ------------------------------------------------------------------
    # Final validation.
    # ------------------------------------------------------------------

    if not repair_verified:
        raise RuntimeError(
            "Foreman failed to perform and verify a successful "
            f"repository edit during repair attempt "
            f"{repair_attempts}/3."
        )

    print("\n=== FOREMAN REPAIR COMPLETE ===")
    print(
        f"Repair attempt {repair_attempts}/3 completed successfully."
    )

    return {
        "foreman_repair_attempts": repair_attempts,
        "phase": "foreman_repair_complete",
    }


def route_after_foreman_review(state: WorkflowState) -> str:
    review = state["foreman_review"]

    if review is None:
        raise RuntimeError("Foreman review missing.")

    if review.decision == "accept":
        return "accepted"

    if review.decision == "repair":
        return "repair"

    if review.decision == "rebuild":
        return "rebuild"

    raise RuntimeError(
        f"Unknown Foreman review decision: {review.decision}"
    )

# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------


def build_graph():
    graph = StateGraph(WorkflowState)

    graph.add_node("planner", planner_node)
    graph.add_node("foreman", foreman_node)
    graph.add_node("worker", worker_node)
    graph.add_node("validator", validator_node)
    graph.add_node("foreman_review", foreman_review_node)
    graph.add_node("foreman_repair", foreman_repair_node)
    graph.add_node("foreman_rebuild", foreman_rebuild_node)

    graph.set_entry_point("planner")

    graph.add_edge("planner", "foreman")
    graph.add_edge("foreman", "worker")
    graph.add_edge("worker", "validator")
    graph.add_edge("validator", "foreman_review")

    graph.add_conditional_edges(
        "foreman_review",
        route_after_foreman_review,
        {
            "accepted": END,
            "repair": "foreman_repair",
            "rebuild": "foreman_rebuild",
        },
    )

    graph.add_edge("foreman_repair", "validator")
    graph.add_edge("foreman_rebuild", "worker")

    return graph.compile()


app = build_graph()


# ---------------------------------------------------------------------------
# Cycle runner
# ---------------------------------------------------------------------------


def run_cycle(
    *,
    cycle_id: int,
    goal: str,
    architecture: str,
    worker_profile: WorkerProfile,
    previous_cycle: CycleResult | None,
) -> CycleResult:
    """
    Execute one complete orchestration cycle.

    This is the Alpha01 memory boundary:
    only CycleResult crosses into the next cycle.
    """

    state = app.invoke(
        {
            "project_goal": goal,
            "architecture_context": architecture,
            "worker_profile": worker_profile,
            "previous_cycle": previous_cycle,
            "plan": None,
            "task_batches": [],
            "task_results": [],
            "worker_attempts": {},
            "foreman_repair_attempts": 0,
            "validation_result": None,
            "foreman_review": None,
            "rework_mode": "none",
            "phase": "initial",
        }
    )

    outcome = (
        "success"
        if all(r.status == "completed" for r in state["task_results"])
        else "partial"
    )

    return CycleResult(
        cycle_id=cycle_id,
        plan=state["plan"],
        task_batches=state["task_batches"],
        task_results=state["task_results"],
        outcome=outcome,
    )


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------


if __name__ == "__main__":

    worker_profile = WorkerProfile(
        model="ibm/granite4.1:3b",
        context_limit="limited",
        strengths=[
            "localized edits",
            "small functions",
            "focused validation",
        ],
        avoid=[
            "architecture redesign",
            "large historical context",
            "unrelated repository areas",
        ],
        execution_mode="one_session_per_batch",
    )

    print("\n==============================")
    print("CYCLE 1")
    print("==============================")

    cycle1 = run_cycle(
        cycle_id=1,
        goal="Design a Python web scraper that extracts a webpage title.",
        architecture="Three-layer orchestration prototype.",
        worker_profile=worker_profile,
        previous_cycle=None,
    )

    print("\n=== CYCLE 1 RESULT ===")
    print(cycle1.model_dump_json(indent=2))

    print("\n==============================")
    print("CYCLE 2")
    print("==============================")

    cycle2 = run_cycle(
        cycle_id=2,
        goal=(
            "Extend the same scraper by improving robustness while "
            "preserving the architecture."
        ),
        architecture="Three-layer orchestration prototype.",
        worker_profile=worker_profile,
        previous_cycle=cycle1,
    )

    print("\n=== CYCLE 2 RESULT ===")
    print(cycle2.model_dump_json(indent=2))
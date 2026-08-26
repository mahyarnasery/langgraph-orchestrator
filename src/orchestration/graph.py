from langgraph.graph import END, StateGraph
from langchain_core.messages import HumanMessage, ToolMessage

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
    read_file,
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

def _foreman_correction_node(state: WorkflowState) -> dict:
    """
    Convert a rejected Foreman review into the next Worker TaskBatch.

    The corrective TaskBatch has already been reasoned about by the Foreman
    review. This function prepares the repository and makes that batch the
    active batch for the next Worker session.
    """

    review = state["foreman_review"]

    if review is None:
        raise RuntimeError("Foreman review missing.")

    batch = review.corrective_batch

    if batch is None:
        raise RuntimeError(
            "Rejected Foreman review has no corrective TaskBatch."
        )

    batch.relevant_files = sorted(
        {
            path
            for task in batch.tasks
            for path in task.relevant_files
        }
    )

    print("\n=== FOREMAN CORRECTIVE BATCH ===")
    print(batch.model_dump_json(indent=2))

    print("\n=== FOREMAN CORRECTIVE REPOSITORY PREPARATION ===")

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
        "task_results": [],
        "review_iteration": state["review_iteration"] + 1,
        "phase": "task_ready",
    }


def foreman_node(state: WorkflowState) -> dict:
    plan = state["plan"]

    if plan is None:
        raise RuntimeError("Planner output missing.")

    review = state["foreman_review"]

    if review is not None and not review.accepted:
        return _foreman_correction_node(state)

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

Create ONE TaskBatch.

The batch may contain multiple related tasks if they share context.

Important:
- Worker edits existing files only.
- Worker never creates files.
- If a task requires a file that does not exist, YOU must prepare/create that file before the Worker executes the task.
- Every file mentioned in any task's relevant_files MUST be included in TaskBatch.relevant_files.
- TaskBatch.relevant_files is the complete union of all task-level relevant_files.
- A task may therefore require creation by the Foreman even though the Worker will only edit the prepared file.

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

    return {
        "task_results": results,
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
    Review the current Worker implementation using both:
    - observed Worker execution evidence
    - actual repository contents

    The Foreman has read-only repository access during review.
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

    review_iteration = state["review_iteration"]

    review_context = "\n\n".join(
        f"TaskResult {result.task_id}:\n"
        f"{result.model_dump_json(indent=2)}"
        for result in results
    )

    relevant_files = sorted(
        set(batch.relevant_files)
        | {
            path
            for result in results
            for action in result.tool_actions
            if action.path
            for path in [action.path]
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

    prompt = f"""
You are the Foreman / Integrator performing an implementation review.

Your responsibility is to determine whether the Worker correctly
implemented the assigned TaskBatch and whether the resulting files form
a coherent implementation.

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

Review iteration:
{review_iteration}

IMPORTANT REVIEW RULES:

1. DETERMINISTIC VALIDATION is authoritative for:
   - syntax
   - file existence
   - scope compliance

2. Do NOT report syntax errors unless they appear in the validator evidence
   or you can directly quote the offending code.

3. Worker TaskResults are execution evidence only.
   Do not assume "completed" means correct.

4. ACTUAL REPOSITORY CONTENTS are the source of truth for implementation.

5. Every reported problem must reference:
   - file
   - function or line (if identifiable)
   - violated acceptance criterion

6. Corrective tasks must address only the reported problems and remain
   within the smallest possible file scope.

7. Do NOT perform a high-level architectural redesign.
   The Planner is responsible for architecture.

8. If implementation is insufficient:
   - Set accepted=false.
   - List concrete implementation problems.
   - Create a NEW corrective TaskBatch.
   - The corrective batch must contain ONLY the work required to fix
     the identified problems.
   - Base corrective tasks on the ACTUAL CURRENT FILE CONTENTS.
   - Do NOT simply repeat the previous task with different wording.
   - Make corrective tasks small and deterministic enough for the
     limited-context Worker.
   - Worker may edit existing files only.
   - If a required file does not exist, the Foreman must prepare it
     before Worker execution.

9. If implementation is sufficient:
   - Set accepted=true.
   - corrective_batch MUST be null.

10. If implementation is NOT sufficient:
   - Set accepted=false.
   - corrective_batch is REQUIRED.
   - Never return accepted=false with corrective_batch=null.
   - Every problem listed must be addressed by at least one corrective task.
   - If only one file needs fixing, create a TaskBatch containing exactly one task.

OUTPUT CONTRACT (mandatory):

IF accepted=true:
    corrective_batch = null

IF accepted=false:
    corrective_batch = valid TaskBatch

This contract is mandatory. Do not violate it.

Return only the structured ForemanReview.
"""

    with model_session(FOREMAN_MODEL) as llm:
        structured = llm.with_structured_output(ForemanReview)
        review = structured.invoke(prompt)

    if not review.accepted and review.corrective_batch is None:
        failed = [
            task
            for task, result in zip(batch.tasks, results)
            if result.status != "completed"
        ]

        if not failed:
            failed = batch.tasks

        repair_batch = TaskBatch(
            batch_id=f"{batch.batch_id}_repair",
            objective="Repair issues identified during Foreman review.",
            tasks=failed,
            shared_context=[],
            relevant_files=sorted({
                f
                for task in failed
                for f in task.relevant_files
            }),
            constraints=[],
            execution_order=[t.task_id for t in failed],
        )

        review.corrective_batch = repair_batch

    print("\n=== FOREMAN REVIEW ===")
    print(review.model_dump_json(indent=2))

    return {
        "foreman_review": review,
        "phase": (
            "foreman_review_accepted"
            if review.accepted
            else "foreman_review_rejected"
        ),
    }


def route_after_foreman_review(state: WorkflowState) -> str:
    """
    Route accepted implementations toward the next Alpha03 stage or send
    rejected implementations back through Foreman correction.
    """

    review = state["foreman_review"]

    if review is None:
        raise RuntimeError("Foreman review missing.")

    if review.accepted:
        return "accepted"

    if review.corrective_batch is None:
        raise RuntimeError(
            "Foreman rejected implementation without a corrective TaskBatch."
        )

    if state["review_iteration"] >= 3:
        raise RuntimeError(
            "Foreman review iteration limit exceeded."
        )

    return "rework"

# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------


def build_graph():
    graph = StateGraph(WorkflowState)

    graph.add_node("planner", planner_node)
    graph.add_node("foreman", foreman_node)
    graph.add_node("validator", validator_node)
    graph.add_node("worker", worker_node)
    graph.add_node("foreman_review", foreman_review_node)

    graph.set_entry_point("planner")

    graph.add_edge("planner", "foreman")
    graph.add_edge("foreman", "worker")
    graph.add_edge("worker", "validator")
    graph.add_edge("validator", "foreman_review")

    graph.add_conditional_edges(
        "foreman_review",
        route_after_foreman_review,
        {
            "rework": "foreman",
            "accepted": END,
        },
    )

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
            "foreman_review": None,
            "review_iteration": 0,
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
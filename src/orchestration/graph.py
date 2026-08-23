from langgraph.graph import END, StateGraph
from langchain_core.messages import HumanMessage, ToolMessage

from .models import (
    FOREMAN_MODEL,
    PLANNER_MODEL,
    WORKER_MODEL,
    model_session,
)
from .schemas import (
    CycleResult,
    Plan,
    Task,
    TaskBatch,
    TaskResult,
    WorkerProfile,
)
from .state import WorkflowState
from .tools import (
    FOREMAN_TOOLS,
    WORKER_TOOLS,
    create_file,
    read_file,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _tool_map(tools):
    return {tool.name: tool for tool in tools}


FOREMAN_TOOL_MAP = _tool_map(FOREMAN_TOOLS)
WORKER_TOOL_MAP = _tool_map(WORKER_TOOLS)


def _task_result_from_response(task: Task, response) -> TaskResult:
    """
    Convert the Worker's final response into compact Alpha01 state.
    """

    return TaskResult(
        task_id=task.task_id,
        status="success",
        actions=["read_file", "edit_file"],
        files_changed=task.relevant_files,
        validation=task.acceptance_criteria,
        errors=[],
    )


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

Create ONE TaskBatch.

The batch may contain multiple related tasks if they share context.

Important:
- Worker edits existing files only.
- Worker never creates files.
- Repository preparation is your responsibility.

Return only the structured TaskBatch.
"""

    with model_session(FOREMAN_MODEL) as llm:
        structured = llm.with_structured_output(TaskBatch)
        batch = structured.invoke(prompt)

        print("\n=== FOREMAN RESULT ===")
        print(batch.model_dump_json(indent=2))

        print("\n=== FOREMAN REPOSITORY PREPARATION ===")

        for relative_path in batch.relevant_files:
            if relative_path in ("", ".", ".."):
                continue

            try:
                read_file.invoke({"path": relative_path})
                print(f"{relative_path} already exists")

            except FileNotFoundError:
                create_file.invoke(
                    {
                        "path": relative_path,
                        "content": "",
                    }
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
- Read existing files
- Edit existing files
- Never create files
- Never redesign architecture
- Output raw source code when editing files
"""

    messages = [HumanMessage(content=prompt)]

    for iteration in range(5):
        print(
            f"\n=== {task.task_id} | iteration {iteration+1}/5 ==="
        )

        response = llm_with_tools.invoke(messages)

        if not response.tool_calls:
            result = _task_result_from_response(task, response)

            print("\n=== TASK RESULT ===")
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
                tool_result = WORKER_TOOL_MAP[name].invoke(
                    tool_call["args"]
                )
            except Exception as exc:
                tool_result = (
                    f"TOOL_ERROR: {type(exc).__name__}: {exc}"
                )

            messages.append(
                ToolMessage(
                    content=str(tool_result),
                    tool_call_id=tool_call["id"],
                )
            )

    return TaskResult(
        task_id=task.task_id,
        status="failure",
        actions=[],
        files_changed=[],
        validation=[],
        errors=["iteration_limit_exceeded"],
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
# Graph
# ---------------------------------------------------------------------------


def build_graph():
    graph = StateGraph(WorkflowState)

    graph.add_node("planner", planner_node)
    graph.add_node("foreman", foreman_node)
    graph.add_node("worker", worker_node)

    graph.set_entry_point("planner")

    graph.add_edge("planner", "foreman")
    graph.add_edge("foreman", "worker")
    graph.add_edge("worker", END)

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
            "phase": "initial",
        }
    )

    outcome = (
        "success"
        if all(r.status == "success" for r in state["task_results"])
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
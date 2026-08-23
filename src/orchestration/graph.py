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
    Convert the Worker's final response into compact Alpha01 task state.

    Alpha01 intentionally stores factual execution metadata instead of prose.
    """

    content = response.content

    if isinstance(content, list):
        text = []
        for item in content:
            if isinstance(item, dict):
                if item.get("text"):
                    text.append(item["text"])
            else:
                text.append(str(item))
        content = "\n".join(text)

    content = str(content).strip()

    return TaskResult(
        task_id=task.task_id,
        status="success",
        actions=["worker_session"],
        files_changed=task.relevant_files,
        validation=task.acceptance_criteria,
        errors=[],
    )


# ---------------------------------------------------------------------------
# Planner
# ---------------------------------------------------------------------------


def planner_node(state: WorkflowState) -> dict:
    """
    Gemma 4 e4b — Planner

    Alpha01 addition:
        - Receives WorkerProfile
        - Receives Previous Cycle
    """

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
    """
    Granite 8B — Foreman

    Commit 1 behavior:
        Creates ONE TaskBatch containing ONE Task.
    """

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

Create ONE focused worker task.

Do not ask the Worker to create files.
"""

    with model_session(FOREMAN_MODEL) as llm:
        structured = llm.with_structured_output(Task)
        task = structured.invoke(prompt)

        batch = TaskBatch(
            batch_id="batch-001",
            objective=task.objective,
            tasks=[task],
            shared_context=[],
            relevant_files=task.relevant_files,
            constraints=task.constraints,
            execution_order=[task.task_id],
        )

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


def worker_node(state: WorkflowState) -> dict:
    """
    Granite 3B — Worker

    Commit 1:
        Executes the first TaskBatch.
        (Currently that batch contains one task.)
    """

    if not state["task_batches"]:
        raise RuntimeError("No TaskBatch received.")

    batch = state["task_batches"][0]
    task = batch.tasks[0]

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
- Stay inside task scope
"""

    messages = [HumanMessage(content=prompt)]

    with model_session(WORKER_MODEL) as llm:
        llm_with_tools = llm.bind_tools(WORKER_TOOLS)

        for iteration in range(5):
            print(f"\n=== Worker iteration {iteration+1}/5 ===")

            response = llm_with_tools.invoke(messages)

            if not response.tool_calls:
                result = _task_result_from_response(task, response)

                print("\n=== TASK RESULT ===")
                print(result.model_dump_json(indent=2))

                return {
                    "task_results": [result],
                    "phase": "worker_complete",
                }

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

    raise RuntimeError("Worker exceeded iteration limit.")


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
# Smoke test
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    from .schemas import WorkerProfile

    result = app.invoke(
        {
            "project_goal": (
                "Design a small Python web scraper that retrieves a webpage "
                "and extracts its title."
            ),
            "architecture_context": (
                "Three-layer orchestration prototype."
            ),
            "worker_profile": WorkerProfile(
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
            ),
            "previous_cycle": None,
            "plan": None,
            "task_batches": [],
            "task_results": [],
            "phase": "initial",
        }
    )

    print("\n==============================")
    print("FINAL WORKFLOW STATE")
    print("==============================")

    print(result["plan"].model_dump_json(indent=2))

    print("\nTask batch:")
    print(result["task_batches"][0].model_dump_json(indent=2))

    print("\nTask result:")
    print(result["task_results"][0].model_dump_json(indent=2))

    cycle = CycleResult(
        cycle_id=1,
        plan=result["plan"],
        task_batches=result["task_batches"],
        task_results=result["task_results"],
        outcome="success",
    )

    print("\nCycleResult:")
    print(cycle.model_dump_json(indent=2))
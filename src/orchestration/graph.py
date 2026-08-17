from langgraph.graph import END, StateGraph
from langchain_core.messages import HumanMessage, ToolMessage

from .models import (
    FOREMAN_MODEL,
    PLANNER_MODEL,
    WORKER_MODEL,
    model_session,
)
from .schemas import Plan, Task, WorkerResult
from .state import WorkflowState
from .tools import (
    FOREMAN_TOOLS,
    WORKER_TOOLS,
    create_file,
    edit_file,
    read_file,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _tool_map(tools):
    """Build a name -> tool mapping from LangChain tools."""
    return {tool.name: tool for tool in tools}


FOREMAN_TOOL_MAP = _tool_map(FOREMAN_TOOLS)
WORKER_TOOL_MAP = _tool_map(WORKER_TOOLS)


def _worker_result_from_response(
    task: Task,
    response,
) -> WorkerResult:
    """
    Convert the worker's final natural-language response into the
    orchestration-level WorkerResult schema.

    IMPORTANT:
    This does not invoke another model.

    The worker's final response is deliberately kept as the implementation
    payload so the orchestration layer does not spend another model
    invocation merely to reformat the result.
    """

    content = response.content

    if isinstance(content, list):
        text_parts = []

        for item in content:
            if isinstance(item, dict):
                text = item.get("text")
                if text:
                    text_parts.append(str(text))
            else:
                text_parts.append(str(item))

        content = "\n".join(text_parts)

    content = str(content).strip()

    if not content:
        content = "Worker completed the task without a textual summary."

    return WorkerResult(
        task_id=task.task_id,
        status="completed",
        summary=content,
        implementation=content,
    )


# ---------------------------------------------------------------------------
# Planner
# ---------------------------------------------------------------------------


def planner_node(state: WorkflowState) -> dict:
    """
    Gemma 4 e4b — Planner / Architect.

    Responsibilities:
    - Understand the overall goal.
    - Produce the high-level implementation plan.
    - Identify affected repository areas.
    - Define acceptance criteria.

    The Planner does NOT modify the repository.
    """

    prompt = f"""
You are the Planner / Architect.

Your responsibility is high-level repository-aware planning.

Do NOT implement the task.
Do NOT write code.
Do NOT create or modify files.

Project goal:
{state["project_goal"]}

Architecture context:
{state["architecture_context"]}

Produce a concise implementation plan.

The plan must identify:
- the objective
- the high-level approach
- affected areas/files
- concrete acceptance criteria

Prefer repository-aware decisions.
If files need to exist, identify them in the plan so that the
Foreman can prepare the repository before delegating implementation
work to the Worker.
"""

    # Exactly ONE model session for the Planner layer.
    with model_session(PLANNER_MODEL) as llm:
        structured_llm = llm.with_structured_output(Plan)
        plan = structured_llm.invoke(prompt)

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
    Granite 4.1 8B — Foreman / Integrator.

    Responsibilities:
    - Turn the Planner's architecture into an executable task.
    - Determine which repository files must exist.
    - Prepare repository structure.
    - Create missing files before the Worker starts.
    - Give the Worker a narrowly scoped implementation task.

    The Foreman is repository-aware and therefore owns file creation.
    """

    plan = state["plan"]

    if plan is None:
        raise RuntimeError("Foreman received no planner output.")

    prompt = f"""
You are the Foreman / Integrator.

Your responsibility is to turn the Planner's architectural plan into
one concrete executable task for a small coding worker.

You are repository-aware.

The Worker is intentionally restricted:
- It may read existing files.
- It may edit existing files.
- It may NOT create files.
- It must not decide that new repository files should exist.

Therefore, you must determine the repository structure required before
delegating the task.

Project goal:
{state["project_goal"]}

Architecture context:
{state["architecture_context"]}

Planner's plan:
{plan.model_dump_json(indent=2)}

Create ONE focused worker task.

The task must:
- have a unique task_id
- be small enough for a limited-context coding worker
- define its scope precisely
- identify relevant files
- state constraints
- define concrete acceptance criteria

If a required file does not exist, include it in relevant_files so that
the orchestration layer can prepare it before the Worker starts.

Do not ask the Worker to create files.
"""

    # Exactly ONE model session for the Foreman layer.
    with model_session(FOREMAN_MODEL) as llm:
        structured_llm = llm.with_structured_output(Task)
        task = structured_llm.invoke(prompt)

        print("\n=== FOREMAN RESULT ===")
        print(task.model_dump_json(indent=2))

        # Repository preparation happens while the Foreman session is alive.
        #
        # The actual file operations are deterministic tools, not additional
        # LLM invocations.
        print("\n=== FOREMAN REPOSITORY PREPARATION ===")

        for relative_path in task.relevant_files:
            if relative_path in ("", ".", ".."):
                continue

            try:
                read_file.invoke({"path": relative_path})

                print(
                    f"=== FOREMAN PREPARATION: "
                    f"{relative_path} already exists ==="
                )

            except FileNotFoundError:
                print(
                    f"=== FOREMAN PREPARATION: "
                    f"creating {relative_path} ==="
                )

                create_file.invoke(
                    {
                        "path": relative_path,
                        "content": "",
                    }
                )

                print(
                    f"=== FOREMAN PREPARATION COMPLETE: "
                    f"{relative_path} ==="
                )

    print("=== FOREMAN REPOSITORY PREPARATION COMPLETE ===")

    return {
        "current_task": task,
        "phase": "task_ready",
    }


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------


def worker_node(state: WorkflowState) -> dict:
    """
    Granite 4.1 3B — Cheap Worker.

    Responsibilities:
    - Execute the narrowly scoped task.
    - Read only what it needs.
    - Modify only existing files.
    - Never create repository files.
    - Never redesign the architecture.

    The Worker gets exactly one model session.

    Multiple tool iterations occur inside that same session.
    """

    task = state["current_task"]

    if task is None:
        raise RuntimeError("Worker received no task.")

    prompt = f"""
You are the Worker.

Your responsibility is to execute ONLY the specific task assigned by
the Foreman.

You have access to filesystem tools operating only inside the worker
workspace.

IMPORTANT WORKSPACE RULES:

1. You may READ existing files.
2. You may EDIT existing files.
3. You may NOT create files.
4. You may NOT create directories.
5. You may NOT modify files outside the assigned task.
6. You may NOT redesign the architecture.
7. Stay strictly within the task scope.
8. If a required file does not exist, report that problem instead of
   attempting to create it.

Project goal:
{state["project_goal"]}

Task:
{task.model_dump_json(indent=2)}

Use the available tools to actually perform the task.

Before editing:
- inspect the relevant existing file(s)
- understand the local context
- make only the changes required by the task

After editing:
- inspect your changes if necessary
- verify that the implementation matches the acceptance criteria

When finished, provide a concise summary of:
- whether the task succeeded
- what was actually changed
- the resulting implementation or important details
"""

    messages = [
        HumanMessage(content=prompt)
    ]

    # Exactly ONE model session for the entire Worker layer.
    with model_session(WORKER_MODEL) as llm:
        llm_with_tools = llm.bind_tools(WORKER_TOOLS)

        for iteration in range(5):
            print(
                f"\n=== Worker iteration "
                f"{iteration + 1}/5 ==="
            )

            response = llm_with_tools.invoke(messages)

            if not response.tool_calls:
                result = _worker_result_from_response(
                    task,
                    response,
                )

                print("\n=== WORKER RESULT ===")
                print(result.model_dump_json(indent=2))

                return {
                    "worker_result": result,
                    "phase": "worker_complete",
                }

            messages.append(response)

            for tool_call in response.tool_calls:
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]

                print(
                    f"\n*** WORKER REQUESTED TOOL: "
                    f"{tool_name}({tool_args}) ***"
                )

                if tool_name not in WORKER_TOOL_MAP:
                    raise RuntimeError(
                        f"Worker requested unauthorized tool: "
                        f"{tool_name}"
                    )

                try:
                    tool_result = WORKER_TOOL_MAP[
                        tool_name
                    ].invoke(tool_args)

                except Exception as exc:
                    tool_result = (
                        f"TOOL_ERROR: "
                        f"{type(exc).__name__}: {exc}"
                    )

                    print(
                        f"*** WORKER TOOL ERROR: "
                        f"{tool_result} ***"
                    )

                else:
                    print(
                        f"*** WORKER TOOL RESULT: "
                        f"{tool_result} ***"
                    )

                messages.append(
                    ToolMessage(
                        content=str(tool_result),
                        tool_call_id=tool_call["id"],
                    )
                )

        raise RuntimeError(
            "Worker exceeded the maximum number of tool iterations."
        )


# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------


def build_graph():
    """
    Construct the minimal three-layer orchestration graph.

    Current workflow:

        Planner
           ↓
        Foreman
           ↓
        repository preparation
           ↓
        Worker
           ↓
          END

    Future architecture can extend this with:
        Worker → Foreman review/integration
               → additional Worker tasks
               → Planner escalation
    """

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
# Direct execution / smoke test
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    result = app.invoke(
        {
            "project_goal": (
                "Design a small Python web scraper that retrieves a "
                "webpage and extracts its title."
            ),
            "architecture_context": (
                "This is a prototype three-layer orchestration workflow. "
                "The Planner is responsible for high-level architecture. "
                "The Foreman is repository-aware and prepares required "
                "files. The Worker is a cheap, limited-context coding "
                "agent that may only read and edit existing files."
            ),
            "plan": None,
            "current_task": None,
            "worker_result": None,
            "phase": "initial",
        }
    )

    print("\n\n==============================")
    print("FINAL WORKFLOW STATE")
    print("==============================")

    print(f"\nPhase: {result['phase']}")

    print("\nPlan:")
    print(result["plan"].model_dump_json(indent=2))

    print("\nTask:")
    print(result["current_task"].model_dump_json(indent=2))

    print("\nWorker result:")
    print(result["worker_result"].model_dump_json(indent=2))
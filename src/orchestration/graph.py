from langgraph.graph import END, StateGraph

from .models import (
    FOREMAN_MODEL,
    PLANNER_MODEL,
    WORKER_MODEL,
    model_session,
)
from .schemas import Plan, Task, WorkerResult
from .state import WorkflowState


def planner_node(state: WorkflowState) -> dict:
    """Gemma: understand the goal and produce an architectural plan."""

    prompt = f"""
You are the Planner / Supervisor.

Your responsibility is high-level understanding and architectural planning.
Do NOT attempt to implement the task directly.

Project goal:
{state["project_goal"]}

Architecture context:
{state["architecture_context"]}

Produce a concise implementation plan.

The plan must identify:
- the objective
- the high-level approach
- affected areas
- acceptance criteria
"""

    with model_session(PLANNER_MODEL) as llm:
        structured_llm = llm.with_structured_output(Plan)
        plan = structured_llm.invoke(prompt)

    print("\n=== PLANNER RESULT ===")
    print(plan.model_dump_json(indent=2))

    return {
        "plan": plan,
        "phase": "planned",
    }


def foreman_node(state: WorkflowState) -> dict:
    """Granite 8B: turn the architectural plan into an executable task."""

    plan = state["plan"]

    if plan is None:
        raise RuntimeError("Foreman received no planner output.")

    prompt = f"""
You are the Foreman / Integrator.

Your responsibility is to turn the planner's architectural plan into
one concrete executable task for a small coding worker.

Do not redesign the architecture.

Project goal:
{state["project_goal"]}

Architecture context:
{state["architecture_context"]}

Planner's plan:
{plan.model_dump_json(indent=2)}

Create ONE focused task.

The task must:
- have a unique task_id
- be small enough for a limited-context coding worker
- define its scope precisely
- identify relevant files or areas
- state constraints
- define concrete acceptance criteria
"""

    with model_session(FOREMAN_MODEL) as llm:
        structured_llm = llm.with_structured_output(Task)
        task = structured_llm.invoke(prompt)

    print("\n=== FOREMAN RESULT ===")
    print(task.model_dump_json(indent=2))

    return {
        "current_task": task,
        "phase": "task_ready",
    }


def worker_node(state: WorkflowState) -> dict:
    """Granite 3B: execute the focused task conceptually."""

    task = state["current_task"]

    if task is None:
        raise RuntimeError("Worker received no task.")

    prompt = f"""
You are the Worker.

Your responsibility is to execute the specific task assigned by the
Foreman.

Do not redesign the overall architecture.
Stay strictly within the task scope.

Project goal:
{state["project_goal"]}

Task:
{task.model_dump_json(indent=2)}

Return a concise implementation result describing:
- whether the task succeeded
- what was implemented
- the resulting implementation
"""

    with model_session(WORKER_MODEL) as llm:
        structured_llm = llm.with_structured_output(WorkerResult)
        result = structured_llm.invoke(prompt)

    print("\n=== WORKER RESULT ===")
    print(result.model_dump_json(indent=2))

    return {
        "worker_result": result,
        "phase": "worker_complete",
    }


def build_graph():
    """Construct the minimal three-layer orchestration graph."""

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


if __name__ == "__main__":
    result = app.invoke(
        {
            "project_goal": (
                "Design a small Python web scraper that retrieves a webpage "
                "and extracts its title."
            ),
            "architecture_context": (
                "This is a prototype orchestration workflow. "
                "The worker should receive a narrowly scoped task rather "
                "than the entire project history."
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
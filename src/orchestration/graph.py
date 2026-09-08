from __future__ import annotations

from langgraph.graph import END, StateGraph

from src.agents.foreman import ForemanAgent
from src.agents.planner import PlannerAgent
from src.harnesses.aider import AiderHarness

from .schemas import (
    CycleResult,
    ForemanReview,
    Plan,
    TaskBatch,
    TaskResult,
    ValidationResult,
    WorkerProfile,
)
from .state import WorkflowState
from .validator import validate_batch

from pathlib import Path

REPO_ROOT = Path("/mnt/ai/langgraph-orchestrator").resolve()
WORKER_WORKSPACE = (REPO_ROOT / "sandbox").resolve()


def _sandbox_repo_path(path: str) -> str:
    """
    Convert a logical TaskBatch path into its physical repository path
    inside the Worker sandbox.
    """

    normalized = path.removeprefix("sandbox/")

    if normalized in ("", ".", ".."):
        raise ValueError(f"Invalid Worker path: {path}")

    return f"sandbox/{normalized}"


# ============================================================================
# Agent / harness construction
# ============================================================================


def build_planner() -> PlannerAgent:
    return PlannerAgent()


def build_foreman() -> ForemanAgent:
    return ForemanAgent()


def build_worker() -> AiderHarness:
    return AiderHarness()


# ============================================================================
# Planner
# ============================================================================


def planner_node(state: WorkflowState) -> dict:
    planner = build_planner()

    plan = planner.run(
        project_goal=state["project_goal"],
        architecture_context=state["architecture_context"],
        worker_profile=state["worker_profile"],
        previous_cycle=state["previous_cycle"],
    )

    if not isinstance(plan, Plan):
        raise TypeError(
            "PlannerAgent.run() must return Plan."
        )

    print("\n=== PLANNER RESULT ===")
    print(plan.model_dump_json(indent=2))

    return {
        "plan": plan,
        "phase": "planned",
    }


# ============================================================================
# Foreman — task decomposition
# ============================================================================


def foreman_node(state: WorkflowState) -> dict:
    plan = state["plan"]

    if plan is None:
        raise RuntimeError(
            "Planner output missing."
        )

    foreman = build_foreman()

    batch = foreman.create_batch(
        project_goal=state["project_goal"],
        architecture_context=state["architecture_context"],
        worker_profile=state["worker_profile"],
        previous_cycle=state["previous_cycle"],
        plan=plan,
    )

    if not isinstance(batch, TaskBatch):
        raise TypeError(
            "ForemanAgent.create_batch() must return TaskBatch."
        )

    # TaskBatch.relevant_files must be the complete union of
    # task-level relevant_files.
    batch.relevant_files = sorted(
        {
            path
            for task in batch.tasks
            for path in task.relevant_files
        }
    )

    print("\n=== FOREMAN RESULT ===")
    print(batch.model_dump_json(indent=2))

    # ------------------------------------------------------------------
    # Repository preparation
    #
    # The Foreman prepares the repository before Worker execution.
    # Workers edit existing files only.
    # ------------------------------------------------------------------

    from .tools import create_file, read_file

    print("\n=== FOREMAN REPOSITORY PREPARATION ===")

    for path in batch.relevant_files:
        if path in ("", ".", ".."):
            continue

        physical_path = _sandbox_repo_path(path)

        try:
            read_file.func(path=physical_path)
            print(f"{physical_path} already exists")

        except FileNotFoundError:
            create_file.func(
                path=physical_path,
                content="",
            )
            print(f"Created {physical_path}")

    return {
        "task_batches": [batch],
        "phase": "task_ready",
    }


# ============================================================================
# Worker / coding harness
# ============================================================================


def worker_node(state: WorkflowState) -> dict:
    if not state["task_batches"]:
        raise RuntimeError(
            "No TaskBatch received."
        )

    batch = state["task_batches"][0]

    worker = build_worker()

    print("\n=== WORKER SESSION START ===")
    print(f"Batch: {batch.batch_id}")

    results = worker.execute(batch)

    if not isinstance(results, list):
        raise TypeError(
            "CodingHarness.execute() must return list[TaskResult]."
        )

    for result in results:
        if not isinstance(result, TaskResult):
            raise TypeError(
                "CodingHarness.execute() returned a non-TaskResult."
            )

    print("\n=== WORKER SESSION COMPLETE ===")

    attempts = dict(state["worker_attempts"])

    for result in results:
        attempts[result.task_id] = (
            attempts.get(result.task_id, 0) + 1
        )

    return {
        "task_results": results,
        "worker_attempts": attempts,
        "phase": "worker_complete",
    }


# ============================================================================
# Validator
# ============================================================================


def validator_node(state: WorkflowState) -> dict:
    if not state["task_batches"]:
        raise RuntimeError(
            "No TaskBatch available for validation."
        )

    validation = validate_batch(
        batch=state["task_batches"][0],
        task_results=state["task_results"],
    )

    if not isinstance(validation, ValidationResult):
        raise TypeError(
            "validate_batch() must return ValidationResult."
        )

    print("\n=== VALIDATOR RESULT ===")
    print(validation.model_dump_json(indent=2))

    return {
        "validation_result": validation,
        "phase": "validated",
    }


# ============================================================================
# Foreman — implementation review
# ============================================================================


def foreman_review_node(state: WorkflowState) -> dict:
    if not state["task_batches"]:
        raise RuntimeError(
            "No TaskBatch available for Foreman review."
        )

    if not state["task_results"]:
        raise RuntimeError(
            "No TaskResults available for Foreman review."
        )

    validation = state["validation_result"]

    if validation is None:
        raise RuntimeError(
            "ValidationResult missing before Foreman review."
        )

    plan = state["plan"]

    if plan is None:
        raise RuntimeError(
            "Planner output missing before Foreman review."
        )

    batch = state["task_batches"][0]

    # Repository inspection is deterministic and remains outside the
    # ForemanAgent. The agent receives the resulting evidence.
    #
    # Files are gathered from:
    #   1. files declared relevant by the Foreman
    #   2. files actually reported as changed by the harness
    #
    # The harness does not expose Aider's internal tool calls.
    relevant_files = sorted(
        set(batch.relevant_files)
        | {
            file_path
            for result in state["task_results"]
            for file_path in result.files_changed
        }
    )

    repository_evidence: list[str] = []

    # Import here so graph.py owns orchestration-time repository
    # inspection without making the agent responsible for it.
    from .tools import read_file

    for path in relevant_files:
        physical_path = _sandbox_repo_path(path)

        try:
            result = read_file.func(path=physical_path)

            repository_evidence.append(
                f"FILE: {path}\n"
                f"{result.get('content', '')}"
            )

        except Exception as exc:
            repository_evidence.append(
                f"FILE: {path}\n"
                f"READ_ERROR: {type(exc).__name__}: {exc}"
            )

        except Exception as exc:
            repository_evidence.append(
                f"FILE: {path}\n"
                f"READ_ERROR: {type(exc).__name__}: {exc}"
            )

    repository_context = "\n\n".join(
        repository_evidence
    )

    foreman = build_foreman()

    review = foreman.review(
        project_goal=state["project_goal"],
        architecture_context=state["architecture_context"],
        plan=plan,
        batch=batch,
        task_results=state["task_results"],
        validation_result=validation,
        repository_context=repository_context,
        worker_attempts=state["worker_attempts"],
    )

    if not isinstance(review, ForemanReview):
        raise TypeError(
            "ForemanAgent.review() must return ForemanReview."
        )

    if review.decision not in {
        "accept",
        "repair",
        "rebuild",
    }:
        raise RuntimeError(
            f"Invalid Foreman review decision: "
            f"{review.decision}"
        )

    if review.decision == "accept":
        review.accepted = True
        review.problems = []
        review.repair_instructions = []
        review.rebuild_task_ids = []

    elif review.decision == "repair":
        review.accepted = False

        if review.rebuild_task_ids:
            raise RuntimeError(
                "Foreman returned rebuild_task_ids for a repair decision."
            )

        if not review.repair_instructions:
            raise RuntimeError(
                "Foreman selected repair without repair_instructions."
            )

    elif review.decision == "rebuild":
        review.accepted = False

        if not review.rebuild_task_ids:
            raise RuntimeError(
                "Foreman selected rebuild without rebuild_task_ids."
            )

        if not review.repair_instructions:
            raise RuntimeError(
                "Foreman selected rebuild without repair_instructions."
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


# ============================================================================
# Routing
# ============================================================================


def route_after_foreman_review(state: WorkflowState) -> str:
    review = state["foreman_review"]

    if review is None:
        raise RuntimeError(
            "Foreman review missing."
        )

    if review.decision == "accept":
        return "accepted"

    if review.decision == "repair":
        return "repair"

    if review.decision == "rebuild":
        return "rebuild"

    raise RuntimeError(
        f"Unsupported Foreman decision: "
        f"{review.decision}"
    )


# ============================================================================
# Graph
# ============================================================================


def build_graph():
    graph = StateGraph(WorkflowState)

    graph.add_node(
        "planner",
        planner_node,
    )

    graph.add_node(
        "foreman",
        foreman_node,
    )

    graph.add_node(
        "worker",
        worker_node,
    )

    graph.add_node(
        "validator",
        validator_node,
    )

    graph.add_node(
        "foreman_review",
        foreman_review_node,
    )

    graph.set_entry_point("planner")

    graph.add_edge(
        "planner",
        "foreman",
    )

    graph.add_edge(
        "foreman",
        "worker",
    )

    graph.add_edge(
        "worker",
        "validator",
    )

    graph.add_edge(
        "validator",
        "foreman_review",
    )

    # Alpha04 Step 4 deliberately stops here.
    #
    # Repair/rebuild execution will be added only after the agent/harness
    # boundary is proven. There is no duplicated Foreman execution logic
    # inside graph.py.

    graph.add_conditional_edges(
        "foreman_review",
        route_after_foreman_review,
        {
            "accepted": END,
            "repair": END,
            "rebuild": END,
        },
    )

    return graph.compile()


app = build_graph()


# ============================================================================
# Cycle runner
# ============================================================================


def run_cycle(
    *,
    cycle_id: int,
    goal: str,
    architecture: str,
    worker_profile: WorkerProfile,
    previous_cycle: CycleResult | None,
) -> CycleResult:

    final_state = app.invoke(
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

    review = final_state["foreman_review"]

    if (
        review is not None
        and review.accepted
        and all(
            result.status == "completed"
            for result in final_state["task_results"]
        )
    ):
        outcome = "overall_success"
    else:
        outcome = "partial"

    return CycleResult(
        cycle_id=cycle_id,
        plan=final_state["plan"],
        task_batches=final_state["task_batches"],
        task_results=final_state["task_results"],
        outcome=outcome,
    )


# ============================================================================
# Smoke test
# ============================================================================


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
        goal=(
            "Design a Python web scraper that extracts "
            "a webpage title."
        ),
        architecture=(
            "Three-layer orchestration prototype."
        ),
        worker_profile=worker_profile,
        previous_cycle=None,
    )

    print("\n=== CYCLE 1 RESULT ===")
    print(cycle1.model_dump_json(indent=2))
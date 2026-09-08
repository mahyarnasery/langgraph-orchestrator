from __future__ import annotations

from pathlib import Path

from langgraph.graph import END, StateGraph

from src.agents.foreman import ForemanAgent
from src.agents.planner import PlannerAgent
from src.harnesses.aider import AiderHarness

from .schemas import (
    CycleResult,
    ForemanReview,
    Plan,
    PlannerReview,
    TaskBatch,
    TaskResult,
    ValidationResult,
    WorkerProfile,
)
from .state import WorkflowState
from .validator import validate_batch

REPO_ROOT = Path("/mnt/ai/langgraph-orchestrator").resolve()
WORKER_WORKSPACE = (REPO_ROOT / "sandbox").resolve()
MAX_FOREMAN_REPAIRS = 2


def _sandbox_repo_path(path: str) -> str:
    """
    Convert a logical TaskBatch path into its physical repository path
    inside the Worker sandbox.
    """
    normalized = path.removeprefix("sandbox/")

    if normalized in ("", ".", ".."):
        raise ValueError(f"Invalid Worker path: {path}")

    return f"sandbox/{normalized}"


def _gather_repository_context(batch: TaskBatch, task_results: list[TaskResult]) -> str:
    """Inspect the sandbox repository files and construct textual evidence."""
    from .tools import read_file

    relevant_files = sorted(
        set(batch.relevant_files)
        | {
            file_path
            for result in task_results
            for file_path in result.files_changed
        }
    )

    repository_evidence: list[str] = []

    for path in relevant_files:
        physical_path = _sandbox_repo_path(path)
        try:
            result = read_file.func(path=physical_path)
            content = result.get("content", "")
            if not content.strip():
                content_display = "[EMPTY FILE - 0 bytes]"
            else:
                content_display = content
            repository_evidence.append(
                f"FILE: {path}\n```\n{content_display}\n```"
            )
        except Exception as exc:
            repository_evidence.append(
                f"FILE: {path}\nREAD_ERROR: {type(exc).__name__}: {exc}"
            )

    return "\n\n".join(repository_evidence)


def _gather_full_repository_context() -> str:
    """Gather context from all Python files in the sandbox for Planner review."""
    from .tools import read_file

    repository_evidence: list[str] = []

    for py_file in WORKER_WORKSPACE.rglob("*.py"):
        try:
            relative_path = py_file.relative_to(WORKER_WORKSPACE)
            logical_path = str(relative_path)
            result = read_file.func(path=f"sandbox/{logical_path}")
            content = result.get("content", "")
            if not content.strip():
                content_display = "[EMPTY FILE - 0 bytes]"
            else:
                content_display = content
            repository_evidence.append(
                f"FILE: {logical_path}\n```\n{content_display}\n```"
            )
        except Exception as exc:
            repository_evidence.append(
                f"FILE: {logical_path}\nREAD_ERROR: {type(exc).__name__}: {exc}"
            )

    return "\n\n".join(repository_evidence) if repository_evidence else "[No Python files in sandbox]"


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
# Planner — Blueprinting
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
# Foreman — Task decomposition & preparation
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

    batch.relevant_files = sorted(
        {
            path
            for task in batch.tasks
            for path in task.relevant_files
        }
    )

    print("\n=== FOREMAN RESULT ===")
    print(batch.model_dump_json(indent=2))

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
            # Create parent directories and empty file
            create_file.func(
                path=physical_path,
                content="",
            )
            print(f"Created {physical_path}")
        except Exception as exc:
            # If read fails for other reasons, still try to create
            print(f"Warning: could not read {physical_path}: {exc}")
            try:
                create_file.func(
                    path=physical_path,
                    content="",
                )
                print(f"Created {physical_path} (after read error)")
            except FileExistsError:
                print(f"{physical_path} already exists (race condition)")
            except Exception as create_exc:
                print(f"Error creating {physical_path}: {create_exc}")

    return {
        "task_batches": [batch],
        "phase": "task_ready",
    }


# ============================================================================
# Worker
# ============================================================================


def worker_node(state: WorkflowState) -> dict:
    if not state["task_batches"]:
        raise RuntimeError(
            "No TaskBatch received."
        )

    batch = state["task_batches"][-1]
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
        attempts[result.task_id] = attempts.get(result.task_id, 0) + 1

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

    batch = state["task_batches"][-1]
    validation = validate_batch(
        batch=batch,
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
# Foreman — Review & Self-Repair
# ============================================================================


def foreman_review_node(state: WorkflowState) -> dict:
    if not state["task_batches"]:
        raise RuntimeError("No TaskBatch available for Foreman review.")
    if not state["task_results"]:
        raise RuntimeError("No TaskResults available for Foreman review.")

    validation = state["validation_result"]
    if validation is None:
        raise RuntimeError("ValidationResult missing before Foreman review.")

    plan = state["plan"]
    if plan is None:
        raise RuntimeError("Planner output missing before Foreman review.")

    batch = state["task_batches"][-1]
    repository_context = _gather_repository_context(batch, state["task_results"])

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
        raise TypeError("ForemanAgent.review() must return ForemanReview.")

    print("\n=== FOREMAN REVIEW ===")
    print(review.model_dump_json(indent=2))

    return {
        "foreman_review": review,
        "phase": f"foreman_review_{review.decision}",
    }


def foreman_self_repair_node(state: WorkflowState) -> dict:
    """Foreman executes direct localized repairs without re-invoking Worker."""
    from .tools import edit_file, write_file

    review = state["foreman_review"]
    print("\n=== FOREMAN DIRECT SELF-REPAIR ===")

    repaired_files: list[str] = []
    for repair in review.direct_repairs:
        physical_path = _sandbox_repo_path(repair.path)
        print(f"Applying Foreman self-repair to {physical_path} ({repair.action})")
        if repair.action == "edit" and repair.target_text:
            edit_file.func(
                path=physical_path,
                old_text=repair.target_text,
                new_text=repair.replacement_text,
            )
        else:
            write_file.func(
                path=physical_path,
                content=repair.content,
            )
        repaired_files.append(repair.path)

    # Record repair as a TaskResult so validation & tracking stay unified
    repair_task_result = TaskResult(
        task_id="foreman_self_repair",
        status="completed",
        summary=f"Foreman direct repair: {', '.join(review.repair_instructions or ['self-repaired'])}",
        files_changed=repaired_files,
        errors=[],
    )

    return {
        "task_results": state["task_results"] + [repair_task_result],
        "foreman_repair_attempts": state["foreman_repair_attempts"] + 1,
        "phase": "foreman_repaired",
    }


def foreman_corrective_node(state: WorkflowState) -> dict:
    """Foreman constructs a new corrective TaskBatch for Worker rebuild."""
    foreman = build_foreman()
    batch = state["task_batches"][-1]
    review = state["foreman_review"]

    print("\n=== FOREMAN GENERATING CORRECTIVE BATCH ===")
    corrective_batch = foreman.create_corrective_batch(
        batch=batch,
        review=review,
        task_results=state["task_results"],
    )

    return {
        "task_batches": state["task_batches"] + [corrective_batch],
        "phase": "corrective_task_ready",
    }


# ============================================================================
# Planner — Acceptance Review
# ============================================================================


def planner_review_node(state: WorkflowState) -> dict:
    """Planner performs the final evaluation against the initial blueprint."""
    planner = build_planner()
    plan = state["plan"]
    batch = state["task_batches"][-1]
    repository_context = _gather_full_repository_context()

    print("\n=== PLANNER FINAL ACCEPTANCE REVIEW ===")
    review = planner.review(
        project_goal=state["project_goal"],
        plan=plan,
        repository_context=repository_context,
    )

    print(review.model_dump_json(indent=2))

    return {
        "planner_review": review,
        "phase": "planner_accepted" if review.accepted else "planner_revision_needed",
    }


# ============================================================================
# Routing
# ============================================================================


def route_after_foreman_review(state: WorkflowState) -> str:
    review = state["foreman_review"]
    if review is None:
        raise RuntimeError("Foreman review missing.")

    if review.decision == "repair":
        if state["foreman_repair_attempts"] >= MAX_FOREMAN_REPAIRS:
            print("Foreman self-repair attempts exceeded limit. Escalating to rebuild.")
            return "rebuild"
        return "self_repair"

    if review.decision == "rebuild":
        return "rebuild"

    if review.decision == "accept":
        return "accepted"

    raise ValueError(f"Unknown Foreman decision: {review.decision}")


# ============================================================================
# Graph construction
# ============================================================================


def build_graph():
    graph = StateGraph(WorkflowState)

    graph.add_node("planner", planner_node)
    graph.add_node("foreman", foreman_node)
    graph.add_node("worker", worker_node)
    graph.add_node("validator", validator_node)
    graph.add_node("foreman_review", foreman_review_node)
    graph.add_node("foreman_self_repair", foreman_self_repair_node)
    graph.add_node("foreman_corrective", foreman_corrective_node)
    graph.add_node("planner_review", planner_review_node)

    graph.set_entry_point("planner")

    graph.add_edge("planner", "foreman")
    graph.add_edge("foreman", "worker")
    graph.add_edge("worker", "validator")
    graph.add_edge("validator", "foreman_review")

    # Routing based on Foreman review
    graph.add_conditional_edges(
        "foreman_review",
        route_after_foreman_review,
        {
            "self_repair": "foreman_self_repair",
            "rebuild": "foreman_corrective",
            "accepted": "planner_review",
        },
    )

    # Direct repairs run through validation again
    graph.add_edge("foreman_self_repair", "validator")

    # Corrective batches are sent to Worker
    graph.add_edge("foreman_corrective", "worker")

    # Final review ends cycle
    graph.add_edge("planner_review", END)

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
            "planner_review": None,

            "rework_mode": "none",
            "phase": "initial",
        }
    )

    foreman_review = final_state.get("foreman_review")
    planner_review = final_state.get("planner_review")

    success = (
        planner_review is not None
        and planner_review.accepted
        and (foreman_review is None or foreman_review.accepted)
    )

    outcome = "overall_success" if success else "partial"

    return CycleResult(
        cycle_id=cycle_id,
        plan=final_state["plan"],
        task_batches=final_state["task_batches"],
        task_results=final_state["task_results"],
        planner_review=planner_review,
        outcome=outcome,
    )

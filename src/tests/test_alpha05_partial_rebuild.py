from pathlib import Path
import subprocess
import sys

import src.orchestration.graph as graph_mod
from src.orchestration.graph import WORKER_WORKSPACE, app
from src.orchestration.ledger import format_ledger_for_prompt
from src.orchestration.schemas import (
    Plan,
    Task,
    TaskBatch,
    TaskResult,
    WorkerProfile,
)


PIPELINE_FILE = WORKER_WORKSPACE / "math_pipeline.py"
MAIN_FILE = WORKER_WORKSPACE / "main.py"
STAGING_DIR = WORKER_WORKSPACE / ".staging"


def main() -> None:
    """
    Deterministic test of Partial Function Rebuild:
    Seeds a 3-function staged module (`math_pipeline.py`) where `t1` (`double_val`)
    and `t3` (`run_pipeline`) are correct, but `t2` (`add_ten`) has a deliberate bug
    (adds 999 instead of 10).
    With `MAX_FOREMAN_REPAIRS = 0`, Foreman must issue a corrective `rebuild` batch
    for `t2`, the 3B Worker rewrites `add_ten` in `.staging/`, and Foreman re-assembles
    all 3 functions into `sandbox/math_pipeline.py`.
    """
    WORKER_WORKSPACE.mkdir(parents=True, exist_ok=True)
    STAGING_DIR.mkdir(parents=True, exist_ok=True)
    for py_file in WORKER_WORKSPACE.rglob("*.py"):
        py_file.unlink(missing_ok=True)

    # Seed staged snippets: t1 (OK), t2 (BUGGY: + 999 instead of + 10), t3 (OK)
    s1 = "def double_val(x: int) -> int:\n    return x * 2\n"
    s2_buggy = "def add_ten(x: int) -> int:\n    return x + 999\n"
    s3 = "def run_pipeline(x: int) -> int:\n    return add_ten(double_val(x))\n"

    (STAGING_DIR / "t1__math_pipeline.py").write_text(s1, encoding="utf-8")
    (STAGING_DIR / "t2__math_pipeline.py").write_text(s2_buggy, encoding="utf-8")
    (STAGING_DIR / "t3__math_pipeline.py").write_text(s3, encoding="utf-8")

    PIPELINE_FILE.write_text(f"{s1}\n\n{s2_buggy}\n\n{s3}", encoding="utf-8")
    MAIN_FILE.write_text(
        "from math_pipeline import run_pipeline\n\n"
        "res = run_pipeline(5)\n"
        "assert res == 20, f'Expected 20 but got {res}'\n"
        "print(f'Pipeline Output: {res}')\n",
        encoding="utf-8",
    )

    plan = Plan(
        objective="Implement math_pipeline.py (double_val, add_ten, run_pipeline) and main.py.",
        approach="Three atomic functions in math_pipeline.py tested by main.py.",
        affected_areas=["math_pipeline.py", "main.py"],
        acceptance_criteria=[
            "double_val(x) returns x * 2.",
            "add_ten(x) returns x + 10.",
            "run_pipeline(x) returns add_ten(double_val(x)).",
            "main.py calls run_pipeline(5), asserts res == 20, and prints 'Pipeline Output: 20'.",
        ],
    )

    initial_batch = TaskBatch(
        batch_id="batch_initial",
        objective=plan.objective,
        tasks=[
            Task(
                task_id="t1",
                objective="Implement double_val(x) returning x * 2",
                target_symbol="double_val",
                scope="math_pipeline.py",
                relevant_files=["math_pipeline.py"],
                constraints=[],
                acceptance_criteria=["double_val(x) returns x * 2"],
            ),
            Task(
                task_id="t2",
                objective="Implement add_ten(x) returning x + 10",
                target_symbol="add_ten",
                scope="math_pipeline.py",
                relevant_files=["math_pipeline.py"],
                constraints=[],
                acceptance_criteria=["add_ten(x) returns x + 10"],
            ),
            Task(
                task_id="t3",
                objective="Implement run_pipeline(x) returning add_ten(double_val(x))",
                target_symbol="run_pipeline",
                scope="math_pipeline.py",
                relevant_files=["math_pipeline.py"],
                dependencies=["t1", "t2"],
                constraints=[],
                acceptance_criteria=["run_pipeline(x) returns add_ten(double_val(x))"],
            ),
            Task(
                task_id="t4",
                objective="Create main.py calling run_pipeline(5) and asserting 20",
                target_symbol=None,
                scope="main.py",
                relevant_files=["main.py"],
                dependencies=["t3"],
                constraints=[],
                acceptance_criteria=["main.py prints 'Pipeline Output: 20'"],
            ),
        ],
        relevant_files=["math_pipeline.py", "main.py"],
        execution_order=["t1", "t2", "t3", "t4"],
    )

    initial_results = [
        TaskResult(
            task_id="t1",
            status="completed",
            summary="Staged t1",
            files_changed=["math_pipeline.py"],
            staged_file=".staging/t1__math_pipeline.py",
        ),
        TaskResult(
            task_id="t2",
            status="completed",
            summary="Staged t2",
            files_changed=["math_pipeline.py"],
            staged_file=".staging/t2__math_pipeline.py",
        ),
        TaskResult(
            task_id="t3",
            status="completed",
            summary="Staged t3",
            files_changed=["math_pipeline.py"],
            staged_file=".staging/t3__math_pipeline.py",
        ),
        TaskResult(
            task_id="t4",
            status="completed",
            summary="Created main.py",
            files_changed=["main.py"],
        ),
    ]

    profile = WorkerProfile(
        model="ibm/granite4.1:3b",
        context_limit="limited",
        strengths=["writing a single isolated Python function at a time"],
        avoid=["writing large multi-function modules in a single task"],
        execution_mode="one_function_per_task_staged",
    )

    # Temporarily set MAX_FOREMAN_REPAIRS = 0 to force the Worker rebuild loop
    orig_max_repairs = graph_mod.MAX_FOREMAN_REPAIRS
    graph_mod.MAX_FOREMAN_REPAIRS = 0

    try:
        # Build a subgraph starting at validator -> foreman_review -> foreman_corrective -> worker -> foreman_assemble -> validator -> ...
        from langgraph.graph import END, StateGraph
        from src.orchestration.state import WorkflowState

        rebuild_graph = StateGraph(WorkflowState)
        rebuild_graph.add_node("validator", graph_mod.validator_node)
        rebuild_graph.add_node("foreman_review", graph_mod.foreman_review_node)
        rebuild_graph.add_node("foreman_corrective", graph_mod.foreman_corrective_node)
        rebuild_graph.add_node("worker", graph_mod.worker_node)
        rebuild_graph.add_node("foreman_assemble", graph_mod.foreman_assemble_node)
        rebuild_graph.add_node("planner_review", graph_mod.planner_review_node)

        rebuild_graph.set_entry_point("validator")
        rebuild_graph.add_edge("validator", "foreman_review")
        rebuild_graph.add_conditional_edges(
            "foreman_review",
            graph_mod.route_after_foreman_review,
            {
                "rebuild": "foreman_corrective",
                "accepted": "planner_review",
            },
        )
        rebuild_graph.add_edge("foreman_corrective", "worker")
        rebuild_graph.add_edge("worker", "foreman_assemble")
        rebuild_graph.add_edge("foreman_assemble", "validator")
        rebuild_graph.add_edge("planner_review", END)

        test_app = rebuild_graph.compile()

        final_state = test_app.invoke(
            {
                "cycle_id": 1,
                "project_goal": (
                    "Create math_pipeline.py with double_val(x) returning x * 2, "
                    "add_ten(x) returning x + 10, and run_pipeline(x) returning "
                    "add_ten(double_val(x)). In main.py, call run_pipeline(5), "
                    "assert res == 20, and print 'Pipeline Output: 20'."
                ),
                "architecture_context": "Staged function assembly in math_pipeline.py.",
                "worker_profile": profile,
                "previous_cycle": None,
                "plan": plan,
                "task_batches": [initial_batch],
                "task_results": initial_results,
                "ledger": [],
                "staged_files": {
                    "math_pipeline.py": {
                        "t1": s1.strip(),
                        "t2": s2_buggy.strip(),
                        "t3": s3.strip(),
                    }
                },
                "file_snapshots": {"math_pipeline.py": "", "main.py": ""},
                "worker_attempts": {"t1": 1, "t2": 1, "t3": 1, "t4": 1},
                "foreman_repair_attempts": 0,
                "rebuild_attempts": 0,
                "validation_result": None,
                "foreman_review": None,
                "planner_review": None,
                "rework_mode": "none",
                "phase": "assembled",
            }
        )
    finally:
        graph_mod.MAX_FOREMAN_REPAIRS = orig_max_repairs

    print("\n==========================================")
    print("=== EXECUTION LEDGER SUMMARY ===")
    print("==========================================")
    print(format_ledger_for_prompt(final_state["ledger"]))

    print("\n=== Re-Assembled math_pipeline.py ===")
    final_code = PIPELINE_FILE.read_text(encoding="utf-8")
    print(final_code)

    # Verify all 3 functions exist and `999` is gone
    for fn in ("def double_val(", "def add_ten(", "def run_pipeline("):
        assert fn in final_code, f"Missing {fn} after partial rebuild!"
    assert "999" not in final_code, "Buggy literal 999 was not replaced during rebuild!"

    completed = subprocess.run(
        [sys.executable, str(MAIN_FILE)],
        cwd=WORKER_WORKSPACE,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, f"main.py failed: {completed.stderr}"
    assert completed.stdout.strip() == "Pipeline Output: 20"
    assert final_state["planner_review"].accepted is True

    print("\n=== ALPHA05 PARTIAL FUNCTION REBUILD TEST PASSED ===")
    print(f"Rebuilds executed: {final_state['rebuild_attempts']}")
    print(f"Output: {completed.stdout.strip()}")


if __name__ == "__main__":
    main()

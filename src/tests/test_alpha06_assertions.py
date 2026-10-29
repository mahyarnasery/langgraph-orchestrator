from pathlib import Path
import subprocess
import sys

from langgraph.graph import END, StateGraph
import src.orchestration.graph as graph_mod
from src.orchestration.graph import WORKER_WORKSPACE, run_project
from src.orchestration.ledger import format_ledger_for_prompt
from src.orchestration.schemas import (
    Plan,
    Task,
    TaskBatch,
    TaskResult,
    WorkerProfile,
)
from src.orchestration.state import WorkflowState
from src.orchestration.validator import validate_batch


STAGING_DIR = WORKER_WORKSPACE / ".staging"


def clean_workspace() -> None:
    WORKER_WORKSPACE.mkdir(parents=True, exist_ok=True)
    STAGING_DIR.mkdir(parents=True, exist_ok=True)
    for py_file in WORKER_WORKSPACE.rglob("*.py"):
        py_file.unlink(missing_ok=True)


def test_part1_deterministic_validator_assertions() -> None:
    """
    Part 1: Verify that `validate_batch` deterministically runs `Task.test_assertions`,
    reports `[actual=..., expected=...]` diagnostics on boundary bugs, identifies
    `failed_task_ids`, checks `target_symbol` via AST, and runs `pytest` on `test_*.py`.
    """
    print("\n============================================================")
    print("=== PART 1: DETERMINISTIC VALIDATOR ASSERTION ENGINE ===")
    print("============================================================")
    clean_workspace()

    grading_file = WORKER_WORKSPACE / "grading.py"
    # clamp_score is correct, but is_passing has a boundary bug (`>` instead of `>=`)
    grading_file.write_text(
        "def clamp_score(val: int, low: int = 0, high: int = 100) -> int:\n"
        "    return max(low, min(high, val))\n\n\n"
        "def is_passing(score: int) -> bool:\n"
        "    return score > 60\n",
        encoding="utf-8",
    )

    batch = TaskBatch(
        batch_id="batch_unit_check",
        objective="Verify grading functions with deterministic unit assertions",
        tasks=[
            Task(
                task_id="t1",
                objective="Implement clamp_score(val, low, high)",
                target_symbol="clamp_score",
                scope="grading.py",
                relevant_files=["grading.py"],
                acceptance_criteria=["`clamp_score(120, 0, 100)` returns `100`"],
                test_assertions=[
                    "assert clamp_score(120, 0, 100) == 100",
                    "assert clamp_score(-5, 0, 100) == 0",
                ],
            ),
            Task(
                task_id="t2",
                objective="Implement is_passing(score) returning score >= 60",
                target_symbol="is_passing",
                scope="grading.py",
                relevant_files=["grading.py"],
                acceptance_criteria=["`is_passing(score)` returns `True` for scores >= 60"],
                test_assertions=[
                    "assert is_passing(75) is True",
                    "assert is_passing(60) is True",
                ],
            ),
        ],
        relevant_files=["grading.py"],
        execution_order=["t1", "t2"],
    )

    results = [
        TaskResult(task_id="t1", status="completed", summary="Done", files_changed=["grading.py"]),
        TaskResult(task_id="t2", status="completed", summary="Done", files_changed=["grading.py"]),
    ]

    val_fail = validate_batch(batch, results)
    print("Validator Result (expecting t2 failure on boundary 60):")
    print(val_fail.model_dump_json(indent=2))

    assert val_fail.passed is False, "Expected validation to fail on boundary assertion `is_passing(60) is True`"
    assert val_fail.failed_task_ids == ["t2"], f"Expected failed_task_ids=['t2'], got {val_fail.failed_task_ids}"
    assert any(
        "assertion failed (t2 [grading.py::is_passing])" in err
        and "actual=False, expected=True" in err
        for err in val_fail.errors
    ), f"Expected actual/expected diagnostic in errors, got: {val_fail.errors}"

    # Fix boundary bug (`score >= 60`) and add a pytest file `test_grading.py`
    grading_file.write_text(
        "def clamp_score(val: int, low: int = 0, high: int = 100) -> int:\n"
        "    return max(low, min(high, val))\n\n\n"
        "def is_passing(score: int) -> bool:\n"
        "    return score >= 60\n",
        encoding="utf-8",
    )
    pytest_file = WORKER_WORKSPACE / "test_grading.py"
    pytest_file.write_text(
        "from grading import clamp_score, is_passing\n\n"
        "def test_grading_suite():\n"
        "    assert clamp_score(105) == 100\n"
        "    assert is_passing(60) is True\n"
        "    assert is_passing(59) is False\n",
        encoding="utf-8",
    )
    batch.relevant_files = ["grading.py", "test_grading.py"]

    val_pass = validate_batch(batch, results)
    print("\nValidator Result after fix + pytest suite:")
    print(val_pass.model_dump_json(indent=2))

    assert val_pass.passed is True, f"Expected validation to pass, got errors: {val_pass.errors}"
    assert val_pass.failed_task_ids == []
    assert any(c.startswith("pytest OK") for c in val_pass.checks), "Expected pytest OK check"
    print("=== PART 1 PASSED ===")


def test_part2_assertion_triggered_surgical_rebuild() -> None:
    """
    Part 2: Seed a 3-function staged module (`grading.py`) + `main.py` where `main.py`
    has NO `assert` and exits with code 0, so ONLY `Task.test_assertions` on `t2`
    (`curve_score(80) == 85`) catches that `t2` added 50 instead of 5.
    Verify that the Validator flags `failed_task_ids=['t2']`, Foreman issues a
    surgical rebuild for `t2`, the Worker rewrites `curve_score`, and re-assembly passes.
    """
    print("\n============================================================")
    print("=== PART 2: SURGICAL REBUILD TRIGGERED BY TASK ASSERTION ===")
    print("============================================================")
    clean_workspace()

    grading_file = WORKER_WORKSPACE / "grading.py"
    main_file = WORKER_WORKSPACE / "main.py"

    s1 = "def base_score(raw: str) -> int:\n    return int(raw.strip())\n"
    s2_buggy = "def curve_score(score: int) -> int:\n    return score + 50\n"
    s3 = "def format_grade(raw: str) -> str:\n    return f'Final Grade: {curve_score(base_score(raw))}'\n"

    (STAGING_DIR / "t1__grading.py").write_text(s1, encoding="utf-8")
    (STAGING_DIR / "t2__grading.py").write_text(s2_buggy, encoding="utf-8")
    (STAGING_DIR / "t3__grading.py").write_text(s3, encoding="utf-8")

    grading_file.write_text(f"{s1}\n\n{s2_buggy}\n\n{s3}", encoding="utf-8")
    # Notice: main.py has NO `assert` statement — it exits 0 even when curve_score is wrong!
    main_file.write_text(
        "from grading import format_grade\n\n"
        "print(format_grade('80'))\n",
        encoding="utf-8",
    )

    plan = Plan(
        objective="Implement grading.py (base_score, curve_score, format_grade) and main.py.",
        approach="Three staged functions in grading.py and entry script main.py.",
        affected_areas=["grading.py", "main.py"],
        acceptance_criteria=[
            "base_score(raw) returns int(raw.strip()).",
            "curve_score(score) returns score + 5.",
            "format_grade(raw) returns f'Final Grade: {curve_score(base_score(raw))}'.",
            "main.py prints 'Final Grade: 85' when calling format_grade('80').",
        ],
    )

    initial_batch = TaskBatch(
        batch_id="batch_grading",
        objective=plan.objective,
        tasks=[
            Task(
                task_id="t1",
                objective="Implement base_score(raw: str) -> int returning int(raw.strip())",
                target_symbol="base_score",
                scope="grading.py",
                relevant_files=["grading.py"],
                acceptance_criteria=["base_score(raw) converts stripped string to int"],
                test_assertions=["assert base_score(' 80 ') == 80"],
            ),
            Task(
                task_id="t2",
                objective="Implement curve_score(score: int) -> int returning score + 5",
                target_symbol="curve_score",
                scope="grading.py",
                relevant_files=["grading.py"],
                acceptance_criteria=["curve_score(score) adds 5 to score"],
                test_assertions=["assert curve_score(80) == 85"],
            ),
            Task(
                task_id="t3",
                objective="Implement format_grade(raw: str) -> str returning f'Final Grade: {curve_score(base_score(raw))}'",
                target_symbol="format_grade",
                scope="grading.py",
                relevant_files=["grading.py"],
                dependencies=["t1", "t2"],
                acceptance_criteria=["format_grade(raw) formats curved score"],
                test_assertions=["assert format_grade('80') == 'Final Grade: 85'"],
            ),
            Task(
                task_id="t4",
                objective="Create main.py printing format_grade('80')",
                target_symbol=None,
                scope="main.py",
                relevant_files=["main.py"],
                dependencies=["t3"],
                acceptance_criteria=["main.py calls format_grade('80')"],
                test_assertions=[],
            ),
        ],
        relevant_files=["grading.py", "main.py"],
        execution_order=["t1", "t2", "t3", "t4"],
    )

    initial_results = [
        TaskResult(
            task_id="t1",
            status="completed",
            summary="Staged t1",
            files_changed=["grading.py"],
            staged_file=".staging/t1__grading.py",
        ),
        TaskResult(
            task_id="t2",
            status="completed",
            summary="Staged t2",
            files_changed=["grading.py"],
            staged_file=".staging/t2__grading.py",
        ),
        TaskResult(
            task_id="t3",
            status="completed",
            summary="Staged t3",
            files_changed=["grading.py"],
            staged_file=".staging/t3__grading.py",
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

    orig_max_repairs = graph_mod.MAX_FOREMAN_REPAIRS
    graph_mod.MAX_FOREMAN_REPAIRS = 0

    try:
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
                    "Create grading.py with base_score(raw) returning int(raw.strip()), "
                    "curve_score(score) returning score + 5, and format_grade(raw) returning "
                    "f'Final Grade: {curve_score(base_score(raw))}'. In main.py, print format_grade('80') "
                    "which must output 'Final Grade: 85'."
                ),
                "architecture_context": "Staged function assembly in grading.py.",
                "worker_profile": profile,
                "previous_cycle": None,
                "plan": plan,
                "task_batches": [initial_batch],
                "task_results": initial_results,
                "ledger": [],
                "staged_files": {
                    "grading.py": {
                        "t1": s1.strip(),
                        "t2": s2_buggy.strip(),
                        "t3": s3.strip(),
                    }
                },
                "file_snapshots": {"grading.py": "", "main.py": ""},
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

    print("\n=== PART 2 LEDGER SUMMARY ===")
    print(format_ledger_for_prompt(final_state["ledger"]))

    final_grading = grading_file.read_text(encoding="utf-8")
    print("\n=== Re-Assembled grading.py ===")
    print(final_grading)

    assert "50" not in final_grading, "Buggy + 50 was not replaced!"
    assert final_state["validation_result"].passed is True
    assert final_state["planner_review"].accepted is True

    completed = subprocess.run(
        [sys.executable, str(main_file)],
        cwd=WORKER_WORKSPACE,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0
    assert completed.stdout.strip() == "Final Grade: 85"
    print("=== PART 2 PASSED ===")


def test_part3_end_to_end_foreman_generated_assertions() -> None:
    """
    Part 3: Full end-to-end orchestration cycle where Foreman generates `TaskBatch`
    with `test_assertions` for a multi-function module (`pricing.py`) and entry script (`main.py`).
    """
    print("\n============================================================")
    print("=== PART 3: END-TO-END FOREMAN ASSERTION GENERATION ===")
    print("============================================================")
    clean_workspace()

    profile = WorkerProfile(
        model="ibm/granite4.1:3b",
        context_limit="limited",
        strengths=["writing a single isolated Python function at a time"],
        avoid=["writing large multi-function modules in a single task"],
        execution_mode="one_function_per_task_staged",
    )

    goal = """
    Create a two-file Python order pricing program in the workspace root:

    1. `pricing.py`:
       Implement three standalone functions in `pricing.py`:
       - `apply_discount(price: int, discount: int) -> int`:
         Returns `price - discount`.
       - `add_shipping(subtotal: int) -> int`:
         Returns `subtotal + 10` if `subtotal < 100` else `subtotal`.
       - `format_invoice(price: int, discount: int) -> str`:
         Calls `apply_discount(price, discount)`, passes that result to `add_shipping(...)` to get `total`,
         and returns `f"Invoice Total: ${total}"`.
       Do not print anything at import time in `pricing.py`.

    2. `main.py`:
       Import `format_invoice` from `pricing`.
       Call `format_invoice(120, 30)`, store the result, assert that it equals `'Invoice Total: $100'`,
       and print the exact result:
       Invoice Total: $100
    """

    architecture = """
    - `pricing.py` has 3 standalone functions (`apply_discount`, `add_shipping`, `format_invoice`) -> split into 3 function-level tasks with `target_symbol` and concrete `test_assertions`.
    - `main.py` is the entry-point script importing `format_invoice` from `pricing`.
    """

    history = run_project(
        goal=goal,
        architecture=architecture,
        worker_profile=profile,
        max_cycles=2,
    )

    final_cycle = history[-1]
    print("\n=== PART 3 EXECUTION LEDGER ===")
    print(format_ledger_for_prompt(final_cycle.ledger))

    initial_batch = final_cycle.task_batches[0]
    pricing_tasks = [
        t for t in initial_batch.tasks if "pricing.py" in t.relevant_files
    ]
    assertions_generated = [
        a for t in pricing_tasks for a in t.test_assertions
    ]
    print(f"\nForeman generated {len(assertions_generated)} task unit assertion(s): {assertions_generated}")
    assert len(assertions_generated) >= 2, (
        f"Expected Foreman to generate unit test_assertions on pricing.py tasks, got {assertions_generated}"
    )

    main_file = WORKER_WORKSPACE / "main.py"
    completed = subprocess.run(
        [sys.executable, str(main_file)],
        cwd=WORKER_WORKSPACE,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, f"main.py failed: {completed.stderr}"
    assert completed.stdout.strip() == "Invoice Total: $100"
    assert final_cycle.outcome == "overall_success"

    print("\n=== ALPHA06 PHASE 1 ALL TESTS PASSED ===")


def main() -> None:
    test_part1_deterministic_validator_assertions()
    test_part2_assertion_triggered_surgical_rebuild()
    test_part3_end_to_end_foreman_generated_assertions()


if __name__ == "__main__":
    main()

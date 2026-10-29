import os
import shutil
import subprocess
import sys

from src.orchestration.checkpoint import (
    CHECKPOINT_FILE,
    load_checkpoint,
    reconcile_checkpoint_with_disk,
    save_checkpoint,
)
from src.orchestration.graph import (
    WORKER_WORKSPACE,
    _prepare_batch_and_workspace,
    run_cycle,
)
from src.orchestration.ledger import format_ledger_for_prompt
from src.orchestration.schemas import (
    Plan,
    Task,
    TaskBatch,
    WorkerProfile,
)


STAGING_DIR = WORKER_WORKSPACE / ".staging"


def clean_workspace() -> None:
    WORKER_WORKSPACE.mkdir(parents=True, exist_ok=True)
    STAGING_DIR.mkdir(parents=True, exist_ok=True)
    for py_file in WORKER_WORKSPACE.rglob("*.py"):
        py_file.unlink(missing_ok=True)
    for child in sorted(WORKER_WORKSPACE.iterdir()):
        if child.is_dir() and child.name != ".git":
            shutil.rmtree(child, ignore_errors=True)


def test_part1_topological_dependency_sorting() -> None:
    """
    Part 1: Verify that `_prepare_batch_and_workspace` topologically sorts
    `batch.execution_order` based on `task.dependencies` and ensures entry-point
    scripts (`main.py`) execute after library tasks even when listed first.
    """
    print("\n================================================================")
    print("=== PART 1: TOPOLOGICAL TASK DEPENDENCY SORTING ===")
    print("================================================================")
    clean_workspace()

    # Deliberately reversed initial execution_order: main (t4) -> summarize (t3) -> add_tax (t2) -> parse_price (t1)
    batch = TaskBatch(
        batch_id="batch_topo",
        objective="Test topological task sorting",
        tasks=[
            Task(
                task_id="t4",
                objective="Create main.py calling summarize_order",
                target_symbol=None,
                scope="main.py",
                relevant_files=["main.py"],
                dependencies=[],  # Even with empty deps, entry script main.py must run after library tasks
                acceptance_criteria=["main.py prints total"],
            ),
            Task(
                task_id="t3",
                objective="Implement summarize_order(raw) calling parse_price and add_tax",
                target_symbol="summarize_order",
                scope="billing.py",
                relevant_files=["billing.py"],
                dependencies=["t1", "t2"],
                acceptance_criteria=["def summarize_order(raw):"],
            ),
            Task(
                task_id="t2",
                objective="Implement add_tax(amount) calling parse_price helper",
                target_symbol="add_tax",
                scope="billing.py",
                relevant_files=["billing.py"],
                dependencies=["t1"],
                acceptance_criteria=["def add_tax(amount):"],
            ),
            Task(
                task_id="t1",
                objective="Implement parse_price(raw)",
                target_symbol="parse_price",
                scope="billing.py",
                relevant_files=["billing.py"],
                dependencies=[],
                acceptance_criteria=["def parse_price(raw):"],
            ),
        ],
        relevant_files=["billing.py", "main.py"],
        execution_order=["t4", "t3", "t2", "t1"],
    )

    prepared, _ = _prepare_batch_and_workspace(batch, {})
    print(f"Initial order:  ['t4', 't3', 't2', 't1']")
    print(f"Sorted order:   {prepared.execution_order}")

    assert prepared.execution_order == ["t1", "t2", "t3", "t4"], (
        f"Expected ['t1', 't2', 't3', 't4'], got {prepared.execution_order}"
    )
    print("=== PART 1 PASSED ===")


def test_part2_mid_worker_interrupt_disk_drift_and_resume() -> None:
    """
    Part 2: Full end-to-end interrupt + disk-reconciled resume test:
    1. Start a cycle with `ORCHESTRATOR_STOP_AFTER_TASKS=2` on a 3-function module + `main.py`.
    2. Catch the simulated KeyboardInterrupt right after 2 tasks finish in Aider.
    3. Verify `sandbox/.checkpoint/state.json` saved the 2 completed tasks.
    4. Simulate disk drift/deletion while paused by deleting the 2nd task's `.staging/` file on disk,
       leaving the 1st task's `.staging/` file intact.
    5. Resume with `run_cycle(..., resume=True)` and verify:
       - Planner and Foreman nodes are skipped completely (0 LLM calls).
       - Task 1 is recognized as valid on disk and skipped by Aider (`worker_attempts[t1] == 1`).
       - Task 2 is detected as missing on disk, invalidated, and re-executed by Aider (`worker_attempts[t2] == 2`).
       - Remaining tasks execute, Foreman assembles the module, Validator passes all assertions,
         and Planner accepts the final project.
    """
    print("\n================================================================")
    print("=== PART 2: MID-WORKER INTERRUPT, DISK DRIFT & RESUME ===")
    print("================================================================")
    clean_workspace()

    profile = WorkerProfile(
        model="ibm/granite4.1:3b",
        context_limit="limited",
        strengths=["writing a single isolated Python function at a time"],
        avoid=["writing large multi-function modules in a single task"],
        execution_mode="one_function_per_task_staged",
    )

    goal = """
    Create a two-file Python metrics calculation program in the workspace root:

    1. `metrics.py`:
       Implement three standalone functions in `metrics.py`:
       - `scale_val(x: int) -> int`:
         Returns `x * 3`.
       - `add_bonus(x: int) -> int`:
         Returns `x + 15`.
       - `compute_metric(x: int) -> str`:
         Calls `scale_val(x)`, passes that result to `add_bonus(...)` to get `total`,
         and returns `f"Metric: {total}"`.
       Do not print anything at import time in `metrics.py`.

    2. `main.py`:
       Import `compute_metric` from `metrics`.
       Call `compute_metric(10)`, store the result, assert that it equals `'Metric: 45'`,
       and print the exact result:
       Metric: 45
    """

    architecture = """
    - `metrics.py` has 3 standalone functions (`scale_val`, `add_bonus`, `compute_metric`) -> split into 3 function-level tasks with `target_symbol` and concrete `test_assertions`.
    - `main.py` is the entry-point script importing `compute_metric` from `metrics`.
    """

    # --- STEP 2A: Run with simulated interrupt after 2 Worker tasks complete ---
    os.environ["ORCHESTRATOR_STOP_AFTER_TASKS"] = "2"
    interrupted = False
    try:
        run_cycle(
            cycle_id=1,
            goal=goal,
            architecture=architecture,
            worker_profile=profile,
            previous_cycle=None,
            resume=False,
        )
    except KeyboardInterrupt as exc:
        interrupted = True
        print(f"\n[TEST] Caught expected mid-batch interrupt: {exc}")
    finally:
        os.environ.pop("ORCHESTRATOR_STOP_AFTER_TASKS", None)

    assert interrupted, "Expected KeyboardInterrupt after 2 tasks!"
    assert CHECKPOINT_FILE.is_file(), f"Expected checkpoint file at {CHECKPOINT_FILE}"

    ckpt_mid = load_checkpoint()
    assert ckpt_mid is not None
    completed_ids_before = [r.task_id for r in ckpt_mid.task_results if r.status == "completed"]
    print(f"[TEST] Checkpointed completed tasks before interrupt: {completed_ids_before}")
    assert len(completed_ids_before) == 2, f"Expected 2 completed tasks in checkpoint, got {completed_ids_before}"

    first_tid, second_tid = completed_ids_before[0], completed_ids_before[1]
    res_map_before = {r.task_id: r for r in ckpt_mid.task_results}
    second_staged_rel = res_map_before[second_tid].staged_file
    assert second_staged_rel, f"Expected task {second_tid} to have a staged_file"

    # --- STEP 2B: Simulate disk corruption/deletion of the 2nd task while paused ---
    second_staged_abs = WORKER_WORKSPACE / second_staged_rel
    print(f"[TEST] Deleting {second_staged_abs.relative_to(WORKER_WORKSPACE)} on disk to simulate disk drift...")
    second_staged_abs.unlink()

    # --- STEP 2C: Resume from checkpoint ---
    print("\n[TEST] Resuming cycle with resume=True...")
    resumed_result = run_cycle(
        cycle_id=1,
        goal=goal,
        architecture=architecture,
        worker_profile=profile,
        previous_cycle=None,
        resume=True,
    )

    print("\n=== PART 2 RESUMED EXECUTION LEDGER ===")
    print(format_ledger_for_prompt(resumed_result.ledger))

    ckpt_final = load_checkpoint()
    assert ckpt_final is not None
    print(f"\n[TEST] Final worker_attempts: {ckpt_final.worker_attempts}")

    # Verify Task 1 was NOT re-run (only 1 attempt total), while deleted Task 2 WAS re-run (2 attempts)
    assert ckpt_final.worker_attempts.get(first_tid) == 1, (
        f"Expected preserved task {first_tid} to have 1 attempt, got {ckpt_final.worker_attempts}"
    )
    assert ckpt_final.worker_attempts.get(second_tid) == 2, (
        f"Expected deleted task {second_tid} to be re-executed (2 attempts), got {ckpt_final.worker_attempts}"
    )

    # Verify Planner and Foreman atomization ran only ONCE (in the first run before interrupt)
    planner_runs = [e for e in resumed_result.ledger if e.layer == "planner" and e.action == "create_plan"]
    foreman_atomize_runs = [e for e in resumed_result.ledger if e.layer == "foreman" and e.action == "atomize_tasks"]
    reconcile_runs = [e for e in resumed_result.ledger if e.layer == "checkpoint" and e.action == "reconcile_disk"]

    assert len(planner_runs) == 1, f"Expected Planner to run only once across interrupt+resume, got {len(planner_runs)}"
    assert len(foreman_atomize_runs) == 1, (
        f"Expected Foreman create_batch to run only once across interrupt+resume, got {len(foreman_atomize_runs)}"
    )
    assert len(reconcile_runs) == 1, "Expected 1 checkpoint disk reconciliation ledger entry"

    # Verify final output of main.py
    main_file = WORKER_WORKSPACE / "main.py"
    completed = subprocess.run(
        [sys.executable, str(main_file)],
        cwd=WORKER_WORKSPACE,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, f"main.py failed:\n{completed.stderr}"
    assert completed.stdout.strip() == "Metric: 45", f"Unexpected output: {completed.stdout!r}"
    assert resumed_result.outcome == "overall_success"
    print("=== PART 2 PASSED ===")


def test_part3_post_assembly_file_deletion_rollback() -> None:
    """
    Part 3: Verify that if a checkpoint is at `phase='assembled'` (or `'validated'`)
    and a user deletes `main.py` on disk, `reconcile_checkpoint_with_disk` detects
    the missing target file, invalidates only `main.py`'s task, and rolls `phase`
    back to `'task_ready'`.
    """
    print("\n================================================================")
    print("=== PART 3: POST-ASSEMBLY DISK DELETION ROLLBACK CHECK ===")
    print("================================================================")

    ckpt = load_checkpoint()
    assert ckpt is not None and ckpt.phase == "planner_accepted"

    # Simulate a checkpoint saved right after assembly (`phase="assembled"`),
    # and delete `sandbox/main.py` on disk.
    ckpt.phase = "assembled"
    main_path = WORKER_WORKSPACE / "main.py"
    assert main_path.is_file()
    main_path.unlink()

    reconciled = reconcile_checkpoint_with_disk(ckpt)
    print(f"Reconciled phase after deleting main.py: {reconciled['phase']}")
    print(f"Remaining verified tasks: {[r.task_id for r in reconciled['task_results']]}")

    assert reconciled["phase"] == "task_ready", (
        f"Expected phase to roll back to 'task_ready' when main.py is deleted, got {reconciled['phase']}"
    )
    verified_ids = {r.task_id for r in reconciled["task_results"]}
    main_tasks = {
        t.task_id
        for t in ckpt.task_batches[-1].tasks
        if "main.py" in t.relevant_files
    }
    assert not (verified_ids & main_tasks), "Expected main.py task to be invalidated after file deletion"
    assert len(verified_ids) >= 3, "Expected all 3 staged tasks for metrics.py to remain verified"

    print("\n=== ALPHA07 ALL CHECKPOINT, TOPOLOGICAL SORT & RESUME TESTS PASSED ===")


def main() -> None:
    test_part1_topological_dependency_sorting()
    test_part2_mid_worker_interrupt_disk_drift_and_resume()
    test_part3_post_assembly_file_deletion_rollback()


if __name__ == "__main__":
    main()

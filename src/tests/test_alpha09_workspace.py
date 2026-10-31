import os
from pathlib import Path
import shutil
import subprocess
import sys

from src.orchestration.checkpoint import get_checkpoint_file, load_checkpoint
import src.orchestration.graph as graph_mod
from src.orchestration.graph import run_staged_project
from src.orchestration.schemas import StageGoal, Task, TaskBatch, TaskResult, WorkerProfile
from src.orchestration.validator import _prepare_assertion_list, validate_batch
from src.orchestration.workspace import (
    configure_access_policy,
    DEFAULT_WORKSPACE_ROOT,
    ORCHESTRATOR_SRC,
    REPO_ROOT,
    get_workspace_root,
    set_active_workspace,
)


CUSTOM_WORKSPACE = Path("/tmp/langgraph_alpha09_custom_app").resolve()


def test_part1_workspace_safety_and_assertion_polish() -> None:
    """
    Part 1: Verify:
    1. `set_active_workspace` blocks unsafe paths (`/`, `REPO_ROOT`, `src/`).
    2. `_prepare_assertion_list` merges standalone setup assignments (`items = [...]`)
       with subsequent `assert` items AND unwraps `assert print(expr) == expected`
       into `assert expr == expected`.
    """
    print("\n================================================================")
    print("=== PART 1: WORKSPACE GUARDS & ASSERTION LIST POLISH ===")
    print("================================================================")

    for forbidden in (Path("/"), REPO_ROOT, ORCHESTRATOR_SRC, ORCHESTRATOR_SRC / "agents"):
        blocked = False
        try:
            set_active_workspace(forbidden)
        except ValueError as exc:
            blocked = True
            print(f"Correctly blocked unsafe workspace '{forbidden}': {exc}")
        assert blocked, f"Expected ValueError for unsafe workspace {forbidden}"

    # Verify multi-item setup + assert merging and `assert print(...)` unwrapping
    raw_assertions = [
        "items = [10, 20, 30]",
        "assert sum(items) == 60",
        "assert print('Hello') == 'Hello'",
    ]
    prepared = _prepare_assertion_list(raw_assertions)
    print(f"Prepared assertions: {prepared}")
    assert prepared == [
        "items = [10, 20, 30]; assert sum(items) == 60",
        "assert 'Hello' == 'Hello'",
    ], f"Unexpected prepared assertions: {prepared}"

    set_active_workspace(DEFAULT_WORKSPACE_ROOT)
    print("=== PART 1 PASSED ===")


def test_part2_custom_external_workspace_staged_build_and_resume() -> None:
    """
    Part 2: Build and resume a 2-stage multi-package OOP application inside an
    EXTERNAL custom directory (`/tmp/langgraph_alpha09_custom_app`), verifying:
    - Default `sandbox/` is completely untouched.
    - `/tmp/langgraph_alpha09_custom_app` gets its own `.staging/`, `.checkpoint/state.json`,
      `domain/sensor.py` (with `class Reading`), `analytics/summary.py`, and `main.py`.
    - Interrupting Stage 2 after 1 task and resuming inside `CUSTOM_WORKSPACE`
      reconciles `/tmp/langgraph_alpha09_custom_app/.checkpoint/state.json` with disk,
      skips Stage 1 and completed Stage 2 Task 1, and finishes the application.
    """
    print("\n================================================================")
    print("=== PART 2: EXTERNAL CUSTOM WORKSPACE BUILD, INTERRUPT & RESUME ===")
    print("================================================================")

    if CUSTOM_WORKSPACE.exists():
        shutil.rmtree(CUSTOM_WORKSPACE, ignore_errors=True)

    # Record a sentinel file in default sandbox/ to prove default sandbox/ is untouched
    set_active_workspace(DEFAULT_WORKSPACE_ROOT)
    sentinel = DEFAULT_WORKSPACE_ROOT / "sentinel_untouched.txt"
    sentinel.write_text("untouched", encoding="utf-8")

    try:
        profile = WorkerProfile(
            model="ibm/granite4.1:3b",
            context_limit="limited",
            strengths=["writing a single isolated Python function or class method at a time"],
            avoid=["writing large multi-file modules in a single task"],
            execution_mode="staged_custom_workspace",
        )

        full_goal = "Build a 2-stage hydrological sensor reading analyzer in the custom workspace."

        stages = [
            StageGoal(
                stage_id=1,
                title="Sensor Reading Domain Class",
                objective=(
                    "Create `domain/sensor.py` defining `class Reading` with two methods:\n"
                    "- `__init__(self, station: str, level_cm: int) -> None`: sets `self.station = station` and `self.level_cm = level_cm`.\n"
                    "- `is_high(self) -> bool`: returns `self.level_cm >= 200`.\n"
                    "Split `class Reading` into 2 method-level tasks (`Reading.__init__`, `Reading.is_high`)."
                ),
                target_files=["domain/sensor.py"],
            ),
            StageGoal(
                stage_id=2,
                title="Hydrological Summary Service & CLI Entry",
                objective=(
                    "Keep `domain/sensor.py` intact and create `analytics/summary.py` and `main.py`:\n"
                    "1. In `analytics/summary.py`, import `Reading` from `domain.sensor` and implement two standalone functions:\n"
                    "   - `count_high(readings: list[Reading]) -> int`: returns `sum(1 for r in readings if r.is_high())`.\n"
                    "   - `format_report(readings: list[Reading]) -> str`: calls `count_high(readings)` and returns `f\"Stations: {len(readings)}, HighWater: {count_high(readings)}\"`.\n"
                    "2. In `main.py`, import `Reading` from `domain.sensor` and `format_report` from `analytics.summary`, "
                    "create `data = [Reading('Dam-A', 250), Reading('Dam-B', 180), Reading('Dam-C', 210)]`, "
                    "assert `format_report(data) == 'Stations: 3, HighWater: 2'`, and print the exact line:\n"
                    "Stations: 3, HighWater: 2"
                ),
                target_files=["analytics/summary.py", "main.py"],
            ),
        ]

        architecture = (
            "Two-stage multi-package Python project in custom workspace: "
            "`domain/sensor.py` (Stage 1), `analytics/summary.py` and `main.py` (Stage 2)."
        )

        orig_run_cycle = graph_mod.run_cycle
        stage2_interrupted = False

        def _hooked_run_cycle(*args, **kwargs):
            nonlocal stage2_interrupted
            goal_str = kwargs.get("goal", "")
            if "Build Stage 2 of 2" in goal_str and not kwargs.get("resume", False) and not stage2_interrupted:
                os.environ["ORCHESTRATOR_STOP_AFTER_TASKS"] = "1"
            try:
                return orig_run_cycle(*args, **kwargs)
            finally:
                os.environ.pop("ORCHESTRATOR_STOP_AFTER_TASKS", None)

        graph_mod.run_cycle = _hooked_run_cycle
        try:
            run_staged_project(
                goal=full_goal,
                architecture=architecture,
                worker_profile=profile,
                stages=stages,
                max_cycles_per_stage=2,
                resume=False,
                workspace=CUSTOM_WORKSPACE,
            )
        except KeyboardInterrupt as exc:
            stage2_interrupted = True
            print(f"\n[TEST] Caught expected mid-Stage-2 interrupt in custom workspace: {exc}")
        finally:
            graph_mod.run_cycle = orig_run_cycle

        assert stage2_interrupted, "Expected KeyboardInterrupt during Stage 2!"
        assert get_workspace_root() == CUSTOM_WORKSPACE
        assert get_checkpoint_file() == CUSTOM_WORKSPACE / ".checkpoint" / "state.json"
        assert get_checkpoint_file().is_file()

        # Verify default sandbox/ did NOT get `domain/sensor.py`
        assert not (DEFAULT_WORKSPACE_ROOT / "domain" / "sensor.py").exists(), (
            "Files leaked into default sandbox/ instead of custom workspace!"
        )
        assert (CUSTOM_WORKSPACE / "domain" / "sensor.py").is_file()

        ckpt_mid = load_checkpoint()
        assert ckpt_mid is not None
        assert ckpt_mid.completed_stage_ids == [1]
        first_stage2_tid = ckpt_mid.task_results[0].task_id

        print(f"\n[TEST] Resuming in custom workspace {CUSTOM_WORKSPACE}...")
        history = run_staged_project(
            goal=full_goal,
            architecture=architecture,
            worker_profile=profile,
            stages=stages,
            max_cycles_per_stage=2,
            resume=True,
            workspace=CUSTOM_WORKSPACE,
        )

        final_cycle = history[-1]
        ckpt_final = load_checkpoint()
        assert ckpt_final is not None
        assert ckpt_final.completed_stage_ids == [1, 2]
        assert ckpt_final.worker_attempts.get(first_stage2_tid) == 1

        for rel_file in ("domain/__init__.py", "domain/sensor.py", "analytics/__init__.py", "analytics/summary.py", "main.py"):
            assert (CUSTOM_WORKSPACE / rel_file).is_file(), f"Missing {rel_file} in {CUSTOM_WORKSPACE}"

        completed = subprocess.run(
            [sys.executable, str(CUSTOM_WORKSPACE / "main.py")],
            cwd=CUSTOM_WORKSPACE,
            capture_output=True,
            text=True,
            check=False,
        )
        assert completed.returncode == 0, f"main.py failed:\n{completed.stderr}"
        output = completed.stdout.strip()
        assert output == "Stations: 3, HighWater: 2", f"Unexpected output: {output!r}"
        assert final_cycle.outcome == "overall_success"
        assert sentinel.read_text(encoding="utf-8") == "untouched"

        print("\n=== ALPHA09 CUSTOM WORKSPACE & MULTI-STAGE RESUME TEST PASSED ===")
        print(f"Custom Workspace: {CUSTOM_WORKSPACE}")
        print(f"Output: {output}")
    finally:
        sentinel.unlink(missing_ok=True)
        set_active_workspace(DEFAULT_WORKSPACE_ROOT)
        if CUSTOM_WORKSPACE.exists():
            shutil.rmtree(CUSTOM_WORKSPACE, ignore_errors=True)


def main() -> None:
    configure_access_policy('allow_external', approve_all_external=True)
    test_part1_workspace_safety_and_assertion_polish()
    test_part2_custom_external_workspace_staged_build_and_resume()


if __name__ == "__main__":
    main()

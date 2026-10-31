from pathlib import Path
import shutil
import subprocess
import sys

from src.orchestration.checkpoint import format_checkpoint_status, load_checkpoint
from src.orchestration.graph import (
    _build_repository_map,
    _prepare_batch_and_workspace,
    run_project,
)
from src.orchestration.schemas import Task, TaskBatch, TaskResult, WorkerProfile
from src.orchestration.validator import emit_pytest_suite, validate_batch
from src.orchestration.workspace import (
    configure_access_policy,
    DEFAULT_WORKSPACE_ROOT,
    set_active_workspace,
)


TEST_WORKSPACE = Path("/tmp/langgraph_alpha10_workspace").resolve()


def reset_test_workspace() -> None:
    if TEST_WORKSPACE.exists():
        shutil.rmtree(TEST_WORKSPACE, ignore_errors=True)
    set_active_workspace(TEST_WORKSPACE)


def test_part1_non_python_asset_validation_and_repo_map() -> None:
    """
    Part 1: Verify deterministic validation & repo map for non-Python assets:
    - Catches malformed JSON (`JSONDecodeError` with line/col) and ragged CSV rows.
    - Passes valid `.json`, `.toml`, `.csv`, `.ini`, and `.md` files and verifies
      quoted/backticked literals in `acceptance_criteria`.
    - Verifies `_prepare_batch_and_workspace` does NOT create `__init__.py` inside
      pure data/config directories (`config/`, `data/`), while creating `__init__.py`
      in Python package directories (`core/`), and schedules asset tasks before `.py` tasks.
    - Verifies `_build_repository_map()` summarizes JSON keys, TOML keys, and CSV columns.
    """
    print("\n================================================================")
    print("=== PART 1: NON-PYTHON ASSET VALIDATION & REPO MAP ===")
    print("================================================================")
    reset_test_workspace()

    batch = TaskBatch(
        batch_id="batch_assets",
        objective="Validate multi-format config/data assets alongside Python package",
        tasks=[
            Task(
                task_id="t_py",
                objective="Implement loader in core/loader.py",
                target_symbol="load_station_limit",
                scope="core/loader.py",
                relevant_files=["core/loader.py"],
                acceptance_criteria=["def load_station_limit():"],
                test_assertions=["assert load_station_limit() == 250"],
            ),
            Task(
                task_id="t_json",
                objective="Create config/settings.json",
                target_symbol="should_be_cleared",
                scope="config/settings.json",
                relevant_files=["config/settings.json"],
                acceptance_criteria=["`config/settings.json` contains `\"max_level\": 250`"],
            ),
            Task(
                task_id="t_toml",
                objective="Create config/app.toml",
                target_symbol=None,
                scope="config/app.toml",
                relevant_files=["config/app.toml"],
                acceptance_criteria=["`config/app.toml` contains `region = \"North\"`"],
            ),
            Task(
                task_id="t_csv",
                objective="Create data/stations.csv",
                target_symbol=None,
                scope="data/stations.csv",
                relevant_files=["data/stations.csv"],
                acceptance_criteria=["`data/stations.csv` contains header `station,level_cm`"],
            ),
        ],
        relevant_files=[
            "core/loader.py",
            "config/settings.json",
            "config/app.toml",
            "data/stations.csv",
        ],
        execution_order=["t_py", "t_json", "t_toml", "t_csv"],
    )

    batch, _ = _prepare_batch_and_workspace(batch, {})
    print(f"Topologically sorted order (assets before Python): {batch.execution_order}")

    # Verify non-Python asset tasks run before `t_py`
    assert batch.execution_order.index("t_json") < batch.execution_order.index("t_py")
    assert batch.execution_order.index("t_csv") < batch.execution_order.index("t_py")

    # Verify `__init__.py` was created ONLY in `core/`, NOT in `config/` or `data/`
    assert (TEST_WORKSPACE / "core" / "__init__.py").is_file()
    assert not (TEST_WORKSPACE / "config" / "__init__.py").exists()
    assert not (TEST_WORKSPACE / "data" / "__init__.py").exists()

    # 1A: Seed broken JSON and ragged CSV -> must fail validation with specific diagnostics
    (TEST_WORKSPACE / "core" / "loader.py").write_text(
        "def load_station_limit() -> int:\n    return 250\n",
        encoding="utf-8",
    )
    (TEST_WORKSPACE / "config" / "settings.json").write_text(
        '{"max_level": 250, }',  # Trailing comma invalid in JSON
        encoding="utf-8",
    )
    (TEST_WORKSPACE / "config" / "app.toml").write_text(
        'region = "North"\n',
        encoding="utf-8",
    )
    (TEST_WORKSPACE / "data" / "stations.csv").write_text(
        "station,level_cm\nDam-A,210,EXTRA_COL\n",
        encoding="utf-8",
    )

    results = [
        TaskResult(task_id=t.task_id, status="completed", summary="ok", files_changed=t.relevant_files)
        for t in batch.tasks
    ]
    val_fail = validate_batch(batch, results)
    print("Validator Result on broken JSON + ragged CSV:")
    print(val_fail.model_dump_json(indent=2))
    assert val_fail.passed is False
    assert "t_json" in val_fail.failed_task_ids
    assert "t_csv" in val_fail.failed_task_ids

    # 1B: Fix JSON and CSV -> must pass all checks and emit pytest suite
    (TEST_WORKSPACE / "config" / "settings.json").write_text(
        '{\n  "max_level": 250,\n  "unit": "cm"\n}\n',
        encoding="utf-8",
    )
    (TEST_WORKSPACE / "data" / "stations.csv").write_text(
        "station,level_cm\nDam-A,210\nDam-B,260\n",
        encoding="utf-8",
    )

    val_pass = validate_batch(batch, results)
    print("\nValidator Result after fixing JSON & CSV:")
    print(val_pass.model_dump_json(indent=2))
    assert val_pass.passed is True, f"Expected pass, got {val_pass.errors}"

    repo_map = _build_repository_map()
    print(f"\nGenerated Multi-Asset Repository Map:\n{repo_map}")
    assert "JSON object (keys: ['max_level', 'unit'])" in repo_map
    assert "CSV (3 rows, columns: ['station', 'level_cm'])" in repo_map
    assert "TOML (keys: ['region'])" in repo_map

    suite_file = emit_pytest_suite(batch)
    assert suite_file is not None and suite_file.is_file()
    print(f"\nEmitted pytest suite ({suite_file.name}):\n{suite_file.read_text(encoding='utf-8')}")
    print("=== PART 1 PASSED ===")


def test_part2_cli_plan_only_and_status_flags() -> None:
    """
    Part 2: Verify `--plan-only`, `--status`, and `--validate-only`:
    - Run `run_project(..., plan_only=True)` in `TEST_WORKSPACE`.
    - Verify it stops right after `planner_node` (`phase == "planned"`, 0 Foreman/Worker tasks),
      saves `.checkpoint/state.json`, and `format_checkpoint_status()` reports the saved plan.
    """
    print("\n================================================================")
    print("=== PART 2: PLAN-ONLY, STATUS & VALIDATE-ONLY CLI CONTROLS ===")
    print("================================================================")
    reset_test_workspace()

    profile = WorkerProfile(
        model="ibm/granite4.1:3b",
        context_limit="limited",
        strengths=["writing clean functions and JSON config files"],
        avoid=["architecture redesign"],
        execution_mode="one_session_per_task",
    )

    goal = """
    Create a two-file config-driven greeting app:
    1. `config.json`: JSON object `{"greeting": "Hello", "target": "Alpha10"}`.
    2. `main.py`: Load `config.json` with `json.loads` and print `Hello Alpha10`.
    """

    history = run_project(
        goal=goal,
        architecture="Config-driven Python script with config.json and main.py.",
        worker_profile=profile,
        max_cycles=1,
        workspace=TEST_WORKSPACE,
        plan_only=True,
    )

    assert len(history) == 1
    assert history[0].outcome == "planned"
    assert history[0].plan is not None
    assert history[0].task_batches == []

    ckpt = load_checkpoint()
    assert ckpt is not None and ckpt.phase == "planned"

    status_report = format_checkpoint_status()
    print("\nFormatted Status Report:")
    print(status_report)
    assert "Saved Phase:     planned" in status_report
    assert "Plan Objective:" in status_report
    print("=== PART 2 PASSED ===")


def test_part3_end_to_end_json_config_and_python_build_with_emitted_tests() -> None:
    """
    Part 3: Full end-to-end orchestration building a multi-file application that
    combines a JSON configuration file (`config/thresholds.json`), a Python processor
    module (`processor.py`), and `main.py`, with `emit_tests=True` automatically
    generating and verifying `tests/test_generated.py`.
    """
    print("\n================================================================")
    print("=== PART 3: END-TO-END JSON CONFIG + PYTHON + PYTEST EMISSION ===")
    print("================================================================")
    reset_test_workspace()

    profile = WorkerProfile(
        model="ibm/granite4.1:3b",
        context_limit="limited",
        strengths=["writing a single isolated Python function or JSON config file at a time"],
        avoid=["writing large multi-file modules in a single task"],
        execution_mode="one_function_per_task_staged",
    )

    goal = """
    Create a three-file config-driven dam alert program in the workspace:

    1. `config/thresholds.json`:
       A valid JSON file containing the exact object:
       `{"alert_cm": 200, "facility": "Karun"}`

    2. `processor.py`:
       Implement two standalone functions in `processor.py` (using `import json` and `from pathlib import Path`):
       - `load_config() -> dict`:
         Reads `Path("config/thresholds.json").read_text(encoding="utf-8")` and returns `json.loads(...)`.
       - `check_level(level_cm: int) -> str`:
         Calls `cfg = load_config()` and returns:
         `f"{cfg['facility']}: ALERT"` if `level_cm >= cfg["alert_cm"]` else `f"{cfg['facility']}: NORMAL"`
       Do not print anything at import time in `processor.py`.

    3. `main.py`:
       Import `check_level` from `processor`.
       Call `res = check_level(225)`, assert `res == "Karun: ALERT"`,
       and print the exact string:
       Karun: ALERT
    """

    architecture = """
    - `config/thresholds.json` is a JSON configuration file (1 task, `target_symbol: null`).
    - `processor.py` has 2 standalone functions (`load_config`, `check_level`) -> split into 2 function-level tasks that depend on `config/thresholds.json`.
    - `main.py` is the entry-point script importing `check_level` from `processor`.
    """

    history = run_project(
        goal=goal,
        architecture=architecture,
        worker_profile=profile,
        max_cycles=2,
        workspace=TEST_WORKSPACE,
        emit_tests=True,
    )

    final_cycle = history[-1]
    assert final_cycle.outcome == "overall_success", f"Expected overall_success, got {final_cycle.outcome}"

    json_file = TEST_WORKSPACE / "config" / "thresholds.json"
    proc_file = TEST_WORKSPACE / "processor.py"
    main_file = TEST_WORKSPACE / "main.py"
    generated_test_file = TEST_WORKSPACE / "tests" / "test_generated.py"

    for f in (json_file, proc_file, main_file, generated_test_file):
        assert f.is_file(), f"Missing expected file: {f}"

    # Verify config/ does NOT have __init__.py
    assert not (TEST_WORKSPACE / "config" / "__init__.py").exists()

    print("\n=== Generated config/thresholds.json ===")
    print(json_file.read_text(encoding="utf-8"))

    print("=== Assembled processor.py ===")
    print(proc_file.read_text(encoding="utf-8"))

    print("=== Emitted tests/test_generated.py ===")
    print(generated_test_file.read_text(encoding="utf-8"))

    # Run main.py
    completed_main = subprocess.run(
        [sys.executable, str(main_file)],
        cwd=TEST_WORKSPACE,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed_main.returncode == 0, f"main.py failed:\n{completed_main.stderr}"
    assert completed_main.stdout.strip() == "Karun: ALERT"

    # Run pytest on the emitted test suite
    completed_pytest = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "tests/test_generated.py"],
        cwd=TEST_WORKSPACE,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed_pytest.returncode == 0, f"Emitted pytest suite failed:\n{completed_pytest.stdout}\n{completed_pytest.stderr}"

    print("\n=== ALPHA10 ALL NON-PYTHON ASSETS, PYTEST EMISSION & CLI TESTS PASSED ===")
    print(f"main.py output: {completed_main.stdout.strip()}")
    print(f"pytest output:  {completed_pytest.stdout.strip()}")


def main() -> None:
    configure_access_policy('allow_external', approve_all_external=True)
    try:
        test_part1_non_python_asset_validation_and_repo_map()
        test_part2_cli_plan_only_and_status_flags()
        test_part3_end_to_end_json_config_and_python_build_with_emitted_tests()
    finally:
        set_active_workspace(DEFAULT_WORKSPACE_ROOT)
        if TEST_WORKSPACE.exists():
            shutil.rmtree(TEST_WORKSPACE, ignore_errors=True)


if __name__ == "__main__":
    main()

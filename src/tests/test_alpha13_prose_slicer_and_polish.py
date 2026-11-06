import shutil
import subprocess
import sys

from src.orchestration.graph import (
    WORKER_WORKSPACE,
    _extract_stage_spec_sections,
    run_staged_project,
)
from src.orchestration.ledger import (
    extract_file_paths_from_goal,
    merge_staged_snippets_ast,
)
from src.orchestration.schemas import Task, TaskBatch, WorkerProfile
from src.orchestration.tools import sanitize_file_content
from src.orchestration.validator import _prepare_assertions, emit_pytest_suite


def clean_workspace() -> None:
    WORKER_WORKSPACE.mkdir(parents=True, exist_ok=True)
    for fpath in WORKER_WORKSPACE.rglob("*"):
        if fpath.is_file() and ".git" not in fpath.parts:
            fpath.unlink(missing_ok=True)
    for child in sorted(WORKER_WORKSPACE.iterdir()):
        if child.is_dir() and child.name != ".git":
            shutil.rmtree(child, ignore_errors=True)


def test_part1_tier1_prose_slicer() -> None:
    """
    Verify that an un-numbered, free-form paragraph specification is sliced into
    self-contained Tier-1 stage specifications without leaking future stages' files
    or requiring the full Tier-0 prose prompt.
    """
    print("\n================================================================")
    print("=== PART 1: TIER-0 -> TIER-1 PROSE SPECIFICATION SLICER ===")
    print("================================================================")

    unnumbered_prose_goal = """
    We are building a solar microgrid telemetry and dispatch platform across multiple packages.
    Start by creating `config/grid_profile.json` with exact JSON keys {"grid_name": "Helios-One", "loss_factor": 0.05, "min_kw": 50.0} along with `data/inverters.csv` containing header inv_id,kw_output,temp_c and three rows:
    INV-01,120.0,45.0
    INV-02,80.0,62.0
    INV-03,40.0,78.0
    In `domain/models.py`, define class `InverterReading` with `__init__(self, inv_id: str, kw_output: float, temp_c: float)` storing those attributes as str, float, and float. Also give `InverterReading` a method `is_viable(self, min_kw: float) -> bool` returning `self.kw_output >= min_kw and self.temp_c < 75.0`.

    For the service layer, create `services/ingestion.py` which imports `InverterReading` from `domain.models` and implements `load_grid_profile(path: str = 'config/grid_profile.json') -> dict` and `load_readings(path: str = 'data/inverters.csv') -> list[InverterReading]`. Next, in `services/analytics.py`, import `InverterReading` from `domain.models` and implement `compute_net_dispatch(readings: list[InverterReading], min_kw: float, loss_factor: float) -> float` which sums `r.kw_output` for readings where `r.is_viable(min_kw)` is True and returns `round(total_kw * (1.0 - loss_factor), 2)`.

    Finally, create `main.py` which loads `config/grid_profile.json` and `data/inverters.csv` via `services.ingestion`, calls `compute_net_dispatch` from `services.analytics`, asserts the result equals `190.0`, and prints `Helios-One Net Dispatch: 190.0 kW`.
    """

    discovered = extract_file_paths_from_goal(unnumbered_prose_goal)
    assert discovered == [
        "config/grid_profile.json",
        "data/inverters.csv",
        "domain/models.py",
        "services/ingestion.py",
        "services/analytics.py",
        "main.py",
    ], f"Unexpected discovered files: {discovered}"

    stage1_files = ["config/grid_profile.json", "data/inverters.csv", "domain/models.py"]
    stage2_files = ["services/ingestion.py", "services/analytics.py"]
    stage3_files = ["main.py"]

    s1_spec = _extract_stage_spec_sections(unnumbered_prose_goal, stage1_files, discovered)
    s2_spec = _extract_stage_spec_sections(unnumbered_prose_goal, stage2_files, discovered)
    s3_spec = _extract_stage_spec_sections(unnumbered_prose_goal, stage3_files, discovered)

    print("--- Stage 1 Tier-1 Spec ---\n" + s1_spec)
    print("\n--- Stage 2 Tier-1 Spec ---\n" + s2_spec)
    print("\n--- Stage 3 Tier-1 Spec ---\n" + s3_spec)

    assert '"Helios-One"' in s1_spec and "INV-01,120.0,45.0" in s1_spec
    assert "InverterReading" in s1_spec and "is_viable" in s1_spec
    assert "services/ingestion.py" not in s1_spec
    assert "compute_net_dispatch" not in s1_spec
    assert "main.py" not in s1_spec

    assert "load_grid_profile" in s2_spec and "load_readings" in s2_spec
    assert "compute_net_dispatch" in s2_spec and "1.0 - loss_factor" in s2_spec
    assert "main.py" not in s2_spec
    assert "INV-01,120.0,45.0" not in s2_spec

    assert "main.py" in s3_spec
    assert "Helios-One Net Dispatch: 190.0 kW" in s3_spec
    assert len(s3_spec) < len(unnumbered_prose_goal) // 2
    print("=== PART 1 PASSED ===")


def test_part2_three_polish_items() -> None:
    """
    Verify the 3 code & test polish items:
    1. `merge_staged_snippets_ast()` normalizes class method indentation via `ast.unparse(ast.parse(merged))`.
    2. `emit_pytest_suite()` only prepends a shared setup line if referenced (`_find_unbound_symbol`),
       and handles multi-line blocks in `main.py` cleanly.
    3. `sanitize_file_content()` strips bare `path/to/...` lines and `_prepare_assertions()` splits `\n`.
    """
    print("\n================================================================")
    print("=== PART 2: THREE CODE & TEST POLISH ITEMS ===")
    print("================================================================")
    clean_workspace()

    s_init = (
        "class Sensor:\n"
        "    def __init__(self, val: float):\n"
        "            self.val = val\n"
    )
    s_scale = (
        "class Sensor:\n"
        "    def scale(self, factor: float) -> float:\n"
        "            return round(self.val * factor, 2)\n"
    )
    merged_cls = merge_staged_snippets_ast(
        "domain/sensor.py",
        "",
        [("t1", s_init), ("t2", s_scale)],
    )
    print("Merged class:\n" + merged_cls)
    assert "        self.val = val" in merged_cls
    assert "            self.val = val" not in merged_cls, "Retained 12-space body indentation!"

    raw_3b_output = "path/to/services/calc.py\ndef double(x: int) -> int:\n    return x * 2\n"
    sanitized = sanitize_file_content("services/calc.py", raw_3b_output)
    assert "path/to/" not in sanitized
    assert sanitized.startswith("def double(x: int) -> int:")

    prepared = _prepare_assertions(
        [
            "readings = [10, 20, 30]\nassert sum(readings) == 60",
            "assert 2 * 3 == 6",
        ]
    )
    assert prepared == [
        "readings = [10, 20, 30]; assert sum(readings) == 60",
        "assert 2 * 3 == 6",
    ], f"Unexpected prepared assertions: {prepared}"

    (WORKER_WORKSPACE / "services").mkdir(parents=True, exist_ok=True)
    (WORKER_WORKSPACE / "services" / "__init__.py").write_text("", encoding="utf-8")
    (WORKER_WORKSPACE / "services" / "calc.py").write_text(
        "def load_items() -> list[int]:\n"
        "    return [2, 4, 6]\n\n\n"
        "def square(x: int) -> int:\n"
        "    return x * x\n",
        encoding="utf-8",
    )
    (WORKER_WORKSPACE / "main.py").write_text(
        "from services.calc import load_items, square\n\n"
        "items = []\n"
        "for v in load_items():\n"
        "    items.append(square(v))\n"
        "assert sum(items) == 56\n"
        "print(f'Total: {sum(items)}')\n",
        encoding="utf-8",
    )

    batch = TaskBatch(
        batch_id="batch_stage_3",
        objective="Test selective setup and test_stage_3 emission",
        tasks=[
            Task(
                task_id="t1",
                objective="Implement load_items",
                target_symbol="load_items",
                scope="services/calc.py",
                relevant_files=["services/calc.py"],
                acceptance_criteria=["def load_items()"],
                test_assertions=["readings = load_items()\nassert len(readings) == 3"],
            ),
            Task(
                task_id="t2",
                objective="Implement square",
                target_symbol="square",
                scope="services/calc.py",
                relevant_files=["services/calc.py"],
                acceptance_criteria=["def square(x)"],
                test_assertions=["assert square(5) == 25"],
            ),
            Task(
                task_id="t3",
                objective="Implement main.py",
                target_symbol=None,
                scope="main.py",
                relevant_files=["main.py"],
                acceptance_criteria=["main.py prints Total: 56"],
                test_assertions=[],
            ),
        ],
        relevant_files=["services/calc.py", "main.py"],
        execution_order=["t1", "t2", "t3"],
    )

    suite_path = emit_pytest_suite(batch, suite_rel_path="tests/test_stage_3.py")
    assert suite_path is not None and suite_path.is_file(), "Failed to emit tests/test_stage_3.py"
    suite_code = suite_path.read_text(encoding="utf-8")
    print("Emitted tests/test_stage_3.py:\n" + suite_code)

    t2_block = suite_code.split("def test_t2_square() -> None:", 1)[1].split("def ", 1)[0]
    assert "readings" not in t2_block, f"Unreferenced setup variable leaked into test_t2_square:\n{t2_block}"
    assert "def test_entry_main() -> None:" in suite_code
    print("=== PART 2 PASSED ===")


def test_part3_end_to_end_unnumbered_prose_multistage() -> None:
    """
    End-to-end 3-stage project execution from an un-numbered, free-form prose specification
    with `--emit-tests` enabled across all stages.
    """
    print("\n================================================================")
    print("=== PART 3: END-TO-END UN-NUMBERED PROSE MULTI-STAGE BUILD ===")
    print("================================================================")
    clean_workspace()

    prose_goal = """
    Build a multi-package solar microgrid telemetry and dispatch system.
    Start by creating `config/grid_profile.json` containing the exact JSON object {"grid_name": "Helios-One", "loss_factor": 0.05, "min_kw": 50.0} along with `data/inverters.csv` containing header inv_id,kw_output,temp_c and exactly 3 data rows:
    INV-01,120.0,45.0
    INV-02,80.0,62.0
    INV-03,40.0,78.0
    In `domain/models.py`, define class `InverterReading` with `__init__(self, inv_id: str, kw_output: float, temp_c: float)` storing `self.inv_id = str(inv_id)`, `self.kw_output = float(kw_output)`, and `self.temp_c = float(temp_c)`, and method `is_viable(self, min_kw: float) -> bool` returning `self.kw_output >= min_kw and self.temp_c < 75.0`.

    Next, in `services/ingestion.py`, import `json`, `csv`, and `InverterReading` from `domain.models`. Implement `load_grid_profile(path: str = 'config/grid_profile.json') -> dict` returning the parsed JSON dictionary, and `load_readings(path: str = 'data/inverters.csv') -> list[InverterReading]` reading the CSV with `csv.DictReader` and returning a list of `InverterReading` instances. In `services/analytics.py`, import `InverterReading` from `domain.models` and implement `compute_net_dispatch(readings: list[InverterReading], min_kw: float, loss_factor: float) -> float` which sums `r.kw_output` for all `r` in `readings` where `r.is_viable(min_kw)` is True and returns `round(total_kw * (1.0 - loss_factor), 2)`.

    Finally, create `main.py` at the workspace root. Import `load_grid_profile` and `load_readings` from `services.ingestion` and `compute_net_dispatch` from `services.analytics`. Load `profile = load_grid_profile('config/grid_profile.json')` and `readings = load_readings('data/inverters.csv')`, compute `net_kw = compute_net_dispatch(readings, profile['min_kw'], profile['loss_factor'])`, assert `net_kw == 190.0`, and print the exact line:
    Helios-One Net Dispatch: 190.0 kW
    """

    profile = WorkerProfile(
        model="ibm/granite4.1:3b",
        context_limit="limited",
        strengths=["localized single-file edits", "writing clean functions, methods, and config files"],
        avoid=["architecture redesign", "interactive input() prompts"],
        execution_mode="one_session_per_task",
    )

    history = run_staged_project(
        goal=prose_goal,
        architecture="Multi-package Python workspace (config/, data/, domain/, services/, main.py).",
        worker_profile=profile,
        max_stages=3,
        max_cycles_per_stage=2,
        emit_tests=True,
    )

    assert history[-1].outcome == "overall_success", f"Expected overall_success, got {history[-1].outcome}"

    completed = subprocess.run(
        [sys.executable, str(WORKER_WORKSPACE / "main.py")],
        cwd=WORKER_WORKSPACE,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, f"main.py failed:\n{completed.stderr}"
    assert completed.stdout.strip() == "Helios-One Net Dispatch: 190.0 kW"

    pytest_run = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "tests"],
        cwd=WORKER_WORKSPACE,
        capture_output=True,
        text=True,
        check=False,
    )
    print("Emitted pytest suite output:\n" + pytest_run.stdout.strip())
    assert pytest_run.returncode == 0, f"Emitted pytest suites failed:\n{pytest_run.stdout}\n{pytest_run.stderr}"
    print("\n=== ALPHA13 ALL TESTS PASSED ===")


def main() -> None:
    test_part1_tier1_prose_slicer()
    test_part2_three_polish_items()
    test_part3_end_to_end_unnumbered_prose_multistage()


if __name__ == "__main__":
    main()

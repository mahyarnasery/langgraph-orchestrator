import shutil
import subprocess
import sys

from src.orchestration.graph import WORKER_WORKSPACE, _prepare_batch_and_workspace, run_project
from src.orchestration.ledger import format_ledger_for_prompt
from src.orchestration.schemas import Task, TaskBatch, TaskResult, WorkerProfile
from src.orchestration.validator import validate_batch


STAGING_DIR = WORKER_WORKSPACE / ".staging"


def clean_workspace() -> None:
    WORKER_WORKSPACE.mkdir(parents=True, exist_ok=True)
    STAGING_DIR.mkdir(parents=True, exist_ok=True)
    for py_file in WORKER_WORKSPACE.rglob("*.py"):
        py_file.unlink(missing_ok=True)
    for child in sorted(WORKER_WORKSPACE.iterdir()):
        if child.is_dir() and child.name not in (".git", ".staging"):
            shutil.rmtree(child, ignore_errors=True)


def test_part1_nested_package_validator_and_relative_imports() -> None:
    """
    Part 1: Verify deterministic package preparation, automatic `__init__.py` creation,
    target_symbol inference, relative import execution (`-m`), and nested `pytest` execution.
    """
    print("\n================================================================")
    print("=== PART 1: NESTED PACKAGE VALIDATION & RELATIVE IMPORTS ===")
    print("================================================================")
    clean_workspace()

    batch = TaskBatch(
        batch_id="batch_pkg_unit",
        objective="Test nested packages core/ and services/ with relative and cross-package imports",
        tasks=[
            Task(
                task_id="t1",
                objective="Implement `clean_tag(raw)` in `core/cleaner.py`",
                target_symbol=None,  # Should be auto-inferred as 'clean_tag'
                scope="Function-level task for core/cleaner.py",
                relevant_files=["core/cleaner.py"],
                acceptance_criteria=["def clean_tag(raw: str) -> str:"],
                test_assertions=["assert clean_tag('  #python ') == 'PYTHON'"],
            ),
            Task(
                task_id="t2",
                objective="Implement `parse_qty(raw)` in `core/cleaner.py`",
                target_symbol=None,  # Should be auto-inferred as 'parse_qty'
                scope="Function-level task for core/cleaner.py",
                relevant_files=["core/cleaner.py"],
                acceptance_criteria=["def parse_qty(raw: str) -> int:"],
                test_assertions=["assert parse_qty(' 42 ') == 42"],
            ),
            Task(
                task_id="t3",
                objective="Implement `format_record` in `services/formatter.py` using relative and package imports",
                target_symbol="format_record",
                scope="services/formatter.py",
                relevant_files=["services/formatter.py"],
                dependencies=["t1", "t2"],
                acceptance_criteria=["format_record combines clean_tag and parse_qty"],
                test_assertions=["assert format_record(' #gpu ', ' 8 ') == 'GPU x8'"],
            ),
        ],
        relevant_files=[
            "core/cleaner.py",
            "services/prefix.py",
            "services/formatter.py",
            "tests/test_pkg.py",
        ],
        execution_order=["t1", "t2", "t3"],
    )

    batch, _ = _prepare_batch_and_workspace(batch, {})

    # Verify automatic target_symbol inference
    assert batch.tasks[0].target_symbol == "clean_tag", f"Expected 'clean_tag', got {batch.tasks[0].target_symbol}"
    assert batch.tasks[1].target_symbol == "parse_qty", f"Expected 'parse_qty', got {batch.tasks[1].target_symbol}"

    # Verify automatic __init__.py creation in nested directories
    for pkg in ("core", "services", "tests"):
        init_path = WORKER_WORKSPACE / pkg / "__init__.py"
        assert init_path.is_file(), f"Expected automatic package marker {init_path}"

    # Write nested module implementations (including a relative import inside services/formatter.py)
    (WORKER_WORKSPACE / "core" / "cleaner.py").write_text(
        "def clean_tag(raw: str) -> str:\n"
        "    return raw.strip().lstrip('#').upper()\n\n\n"
        "def parse_qty(raw: str) -> int:\n"
        "    return int(raw.strip())\n",
        encoding="utf-8",
    )
    (WORKER_WORKSPACE / "services" / "prefix.py").write_text(
        "SEP = ' x'\n",
        encoding="utf-8",
    )
    (WORKER_WORKSPACE / "services" / "formatter.py").write_text(
        "from core.cleaner import clean_tag, parse_qty\n"
        "from .prefix import SEP\n\n\n"
        "def format_record(tag: str, qty: str) -> str:\n"
        "    return f'{clean_tag(tag)}{SEP}{parse_qty(qty)}'\n",
        encoding="utf-8",
    )
    (WORKER_WORKSPACE / "tests" / "test_pkg.py").write_text(
        "from services.formatter import format_record\n\n"
        "def test_format_record():\n"
        "    assert format_record(' #ai ', ' 4 ') == 'AI x4'\n",
        encoding="utf-8",
    )

    results = [
        TaskResult(task_id="t1", status="completed", summary="ok", files_changed=["core/cleaner.py"]),
        TaskResult(task_id="t2", status="completed", summary="ok", files_changed=["core/cleaner.py"]),
        TaskResult(task_id="t3", status="completed", summary="ok", files_changed=["services/formatter.py"]),
    ]

    val = validate_batch(batch, results)
    print("Validator Result on multi-directory package:")
    print(val.model_dump_json(indent=2))

    assert val.passed is True, f"Expected package validation to pass, got errors: {val.errors}"
    assert any("pytest OK (tests/test_pkg.py)" in c for c in val.checks)
    print("=== PART 1 PASSED ===")


def test_part2_end_to_end_multipackage_staged_orchestration() -> None:
    """
    Part 2: Full end-to-end orchestration cycle creating a multi-directory package project:
    - `core/text_utils.py` (2 staged functions: `clean_label`, `parse_value`)
    - `services/report.py` (2 staged functions: `format_entry`, `build_summary` importing from `core.text_utils`)
    - `main.py` (entry point importing `build_summary` from `services.report`)
    """
    print("\n================================================================")
    print("=== PART 2: END-TO-END MULTI-PACKAGE STAGED ORCHESTRATION ===")
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
    Create a multi-directory Python package project inside the workspace:

    1. `core/text_utils.py`:
       Implement two standalone functions in `core/text_utils.py`:
       - `clean_label(raw: str) -> str`:
         Returns `raw.strip().upper()`.
       - `parse_value(raw: str) -> int`:
         Returns `int(raw.strip())`.
       Do not print anything at import time in `core/text_utils.py`.

    2. `services/report.py`:
       Import `clean_label` and `parse_value` from `core.text_utils`.
       Implement two standalone functions in `services/report.py`:
       - `format_entry(line: str) -> tuple[str, int]`:
         Splits `line` on `'='` (`left, right = line.split('=')`) and returns
         `(clean_label(left), parse_value(right))`.
       - `build_summary(lines: list[str]) -> str`:
         Calls `format_entry(line)` for each `line` in `lines` to build a list `entries`,
         and returns `f"Items: {len(entries)}, Sum: {sum(v for _, v in entries)}"`.
       Do not print anything at import time in `services/report.py`.

    3. `main.py`:
       Import `build_summary` from `services.report`.
       Call `build_summary([" cpu = 40 ", " gpu = 60 ", " ram = 25 "])`,
       store the result, assert that it equals `'Items: 3, Sum: 125'`,
       and print the exact result:
       Items: 3, Sum: 125
    """

    architecture = """
    Multi-package Python workspace in `sandbox/`:
    - `core/text_utils.py` has 2 standalone functions (`clean_label`, `parse_value`) -> decompose into 2 function-level tasks with `relevant_files: ["core/text_utils.py"]`.
    - `services/report.py` has 2 standalone functions (`format_entry`, `build_summary`) importing from `core.text_utils` -> decompose into 2 function-level tasks with `relevant_files: ["services/report.py"]`.
    - `main.py` at workspace root imports `build_summary` from `services.report`.
    """

    history = run_project(
        goal=goal,
        architecture=architecture,
        worker_profile=profile,
        max_cycles=2,
    )

    final_cycle = history[-1]
    print("\n=== PART 2 EXECUTION LEDGER ===")
    print(format_ledger_for_prompt(final_cycle.ledger))

    # 1. Verify nested files and __init__.py markers exist
    core_init = WORKER_WORKSPACE / "core" / "__init__.py"
    services_init = WORKER_WORKSPACE / "services" / "__init__.py"
    text_utils_file = WORKER_WORKSPACE / "core" / "text_utils.py"
    report_file = WORKER_WORKSPACE / "services" / "report.py"
    main_file = WORKER_WORKSPACE / "main.py"

    for p in (core_init, services_init, text_utils_file, report_file, main_file):
        assert p.is_file(), f"Missing expected package file: {p}"

    # 2. Verify staged files were created with flattened package names
    staged_core = sorted(f.name for f in STAGING_DIR.glob("*__core__text_utils.py"))
    staged_services = sorted(f.name for f in STAGING_DIR.glob("*__services__report.py"))
    print(f"\nStaged core/text_utils.py snippets: {staged_core}")
    print(f"Staged services/report.py snippets: {staged_services}")
    assert len(staged_core) >= 2, f"Expected >=2 staged snippets for core/text_utils.py, got {staged_core}"
    assert len(staged_services) >= 2, f"Expected >=2 staged snippets for services/report.py, got {staged_services}"

    # 3. Inspect assembled nested modules
    print("\n=== Assembled core/text_utils.py ===")
    print(text_utils_file.read_text(encoding="utf-8"))

    print("=== Assembled services/report.py ===")
    print(report_file.read_text(encoding="utf-8"))

    print("=== Final main.py ===")
    print(main_file.read_text(encoding="utf-8"))

    # 4. Verify runtime output
    completed = subprocess.run(
        [sys.executable, str(main_file)],
        cwd=WORKER_WORKSPACE,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, f"main.py failed:\n{completed.stderr}"
    output = completed.stdout.strip()
    assert output == "Items: 3, Sum: 125", f"Expected 'Items: 3, Sum: 125', got {output!r}"
    assert final_cycle.outcome == "overall_success"

    print("\n=== ALPHA06 PHASE 2 MULTI-PACKAGE ORCHESTRATION TEST PASSED ===")
    print(f"Output: {output}")


def main() -> None:
    test_part1_nested_package_validator_and_relative_imports()
    test_part2_end_to_end_multipackage_staged_orchestration()


if __name__ == "__main__":
    main()

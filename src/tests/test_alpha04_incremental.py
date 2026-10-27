from pathlib import Path
import subprocess
import sys

from src.orchestration.graph import run_cycle
from src.orchestration.schemas import WorkerProfile


REPO_ROOT = Path("/mnt/ai/langgraph-orchestrator").resolve()
WORKSPACE_ROOT = (REPO_ROOT / "sandbox").resolve()

MATH_FILE = WORKSPACE_ROOT / "math_utils.py"
APP_FILE = WORKSPACE_ROOT / "app.py"


def main() -> None:
    # Clean sandbox before Stage 1
    WORKSPACE_ROOT.mkdir(parents=True, exist_ok=True)
    for py_file in WORKSPACE_ROOT.rglob("*.py"):
        py_file.unlink(missing_ok=True)

    profile = WorkerProfile(
        model="ibm/granite4.1:3b",
        context_limit="limited",
        strengths=["localized single-file edits", "writing simple functions"],
        avoid=["architecture redesign", "editing multiple files in one task"],
        execution_mode="one_session_per_task",
    )

    # --------------------------------------------------------
    # STAGE 1: Initial creation of math_utils.py and app.py
    # --------------------------------------------------------
    print("\n==========================================")
    print("=== STAGE 1: INITIAL PROJECT CREATION ===")
    print("==========================================")

    goal_stage_1 = """
    Create a two-file Python program in the workspace root:

    1. `math_utils.py`:
       Define two functions:
       - `add(a, b)` that returns `a + b`
       - `multiply(a, b)` that returns `a * b`
       Do not print anything at import time in `math_utils.py`.

    2. `app.py`:
       Import `add` and `multiply` from `math_utils`.
       Compute `multiply(add(3, 4), 5)` and print the exact line:
       Result: 35
    """

    cycle_1 = run_cycle(
        cycle_id=1,
        goal=goal_stage_1,
        architecture="Two-module Python project: math_utils.py (library) and app.py (entry point).",
        worker_profile=profile,
        previous_cycle=None,
    )

    if cycle_1.outcome != "overall_success":
        raise AssertionError(f"Stage 1 failed with outcome: {cycle_1.outcome}")

    # --------------------------------------------------------
    # STAGE 2: Incremental feature addition on existing codebase
    # --------------------------------------------------------
    print("\n==========================================")
    print("=== STAGE 2: INCREMENTAL MODIFICATION ===")
    print("==========================================")

    goal_stage_2 = """
    Update the existing project in the workspace root without removing existing functions:

    1. In `math_utils.py`:
       Keep `add(a, b)` and `multiply(a, b)` intact, and add a new function:
       - `subtract(a, b)` that returns `a - b`

    2. In `app.py`:
       Use `add`, `multiply`, and `subtract` from `math_utils` to compute
       `subtract(multiply(add(3, 4), 5), 10)` and print the exact line:
       Result: 25
    """

    cycle_2 = run_cycle(
        cycle_id=2,
        goal=goal_stage_2,
        architecture="Extend existing math_utils.py and update app.py calculation.",
        worker_profile=profile,
        previous_cycle=None,
    )

    print("\n=== STAGE 2 CYCLE RESULT ===")
    print(cycle_2.model_dump_json(indent=2))

    print(f"\n=== Final Content of {MATH_FILE.name} ===")
    math_content = MATH_FILE.read_text(encoding="utf-8")
    print(math_content)

    print(f"=== Final Content of {APP_FILE.name} ===")
    print(APP_FILE.read_text(encoding="utf-8"))

    # Verify all three functions still exist in math_utils.py
    for fn_name in ("def add(", "def multiply(", "def subtract("):
        if fn_name not in math_content:
            raise AssertionError(
                f"Incremental edit lost or missed '{fn_name}' in math_utils.py!"
            )

    # Verify execution output of updated app.py
    completed = subprocess.run(
        [sys.executable, str(APP_FILE)],
        cwd=WORKSPACE_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    if completed.returncode != 0:
        raise AssertionError(f"Updated app.py failed to execute:\n{completed.stderr}")

    actual_output = completed.stdout.strip()
    if actual_output != "Result: 25":
        raise AssertionError(
            f"Unexpected updated app.py output: {actual_output!r} (expected 'Result: 25')"
        )

    if cycle_2.outcome != "overall_success":
        raise AssertionError(
            f"Expected Stage 2 outcome 'overall_success', got {cycle_2.outcome!r}"
        )

    print("\n=== ALPHA04 INCREMENTAL TEST PASSED ===")
    print(f"Preserved & extended functions in {MATH_FILE.name}: add, multiply, subtract")
    print(f"Updated output from {APP_FILE.name}: {actual_output}")


if __name__ == "__main__":
    main()

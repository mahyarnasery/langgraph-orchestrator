from pathlib import Path
import subprocess
import sys

from src.orchestration.graph import run_cycle
from src.orchestration.schemas import WorkerProfile


REPO_ROOT = Path("/mnt/ai/langgraph-orchestrator").resolve()
WORKSPACE_ROOT = (REPO_ROOT / "sandbox").resolve()

MATH_FILE = WORKSPACE_ROOT / "math_utils.py"
APP_FILE = WORKSPACE_ROOT / "app.py"

ROOT_MATH_FILE = REPO_ROOT / "math_utils.py"
ROOT_APP_FILE = REPO_ROOT / "app.py"


def main() -> None:
    # --------------------------------------------------------
    # Clean previous sandbox artifacts so repo map starts clean
    # --------------------------------------------------------
    WORKSPACE_ROOT.mkdir(parents=True, exist_ok=True)
    for py_file in WORKSPACE_ROOT.rglob("*.py"):
        py_file.unlink(missing_ok=True)

    ROOT_MATH_FILE.unlink(missing_ok=True)
    ROOT_APP_FILE.unlink(missing_ok=True)

    profile = WorkerProfile(
        model="ibm/granite4.1:3b",
        context_limit="limited",
        strengths=["localized single-file edits", "writing simple functions"],
        avoid=["architecture redesign", "editing multiple files in one task"],
        execution_mode="one_session_per_task",
    )

    goal = """
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

    result = run_cycle(
        cycle_id=1,
        goal=goal,
        architecture="Two-module Python project: math_utils.py (library) and app.py (entry point).",
        worker_profile=profile,
        previous_cycle=None,
    )

    print("\n=== CYCLE RESULT ===")
    print(result.model_dump_json(indent=2))

    # --------------------------------------------------------
    # Isolation & existence assertions
    # --------------------------------------------------------
    if not MATH_FILE.is_file():
        raise AssertionError(f"Expected file was not created: {MATH_FILE}")

    if not APP_FILE.is_file():
        raise AssertionError(f"Expected file was not created: {APP_FILE}")

    if ROOT_MATH_FILE.exists() or ROOT_APP_FILE.exists():
        raise AssertionError("Worker escaped the sandbox and created files in REPO_ROOT!")

    print(f"\n=== Content of {MATH_FILE.name} ===")
    print(MATH_FILE.read_text(encoding="utf-8"))

    print(f"=== Content of {APP_FILE.name} ===")
    print(APP_FILE.read_text(encoding="utf-8"))

    # --------------------------------------------------------
    # Verify actual execution behavior of app.py
    # --------------------------------------------------------
    completed = subprocess.run(
        [sys.executable, str(APP_FILE)],
        cwd=WORKSPACE_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    if completed.returncode != 0:
        raise AssertionError(f"app.py failed to execute:\n{completed.stderr}")

    actual_output = completed.stdout.strip()
    if actual_output != "Result: 35":
        raise AssertionError(
            f"Unexpected app.py output: {actual_output!r} (expected 'Result: 35')"
        )

    if result.outcome != "overall_success":
        raise AssertionError(
            f"Expected cycle outcome 'overall_success', got {result.outcome!r}"
        )

    print("\n=== ALPHA04 MULTI-FILE TEST PASSED ===")
    print(f"Files verified: {MATH_FILE.name}, {APP_FILE.name}")
    print(f"Output: {actual_output}")


if __name__ == "__main__":
    main()

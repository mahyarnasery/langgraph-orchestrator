from pathlib import Path
import subprocess
import sys

from src.orchestration.graph import run_project
from src.orchestration.schemas import WorkerProfile


REPO_ROOT = Path("/mnt/ai/langgraph-orchestrator").resolve()
WORKSPACE_ROOT = (REPO_ROOT / "sandbox").resolve()
TARGET_FILE = WORKSPACE_ROOT / "buggy_calc.py"


def main() -> None:
    WORKSPACE_ROOT.mkdir(parents=True, exist_ok=True)
    for py_file in WORKSPACE_ROOT.rglob("*.py"):
        py_file.unlink(missing_ok=True)

    # Seed sandbox with a broken Python file (1 SyntaxError + 1 operator precedence bug)
    broken_code = """def average(a, b)
    return a + b / 2

result = average(10, 20)
assert result == 15.0, f"Expected 15.0 but got {result}"
print(f"Average: {result}")
"""
    TARGET_FILE.write_text(broken_code, encoding="utf-8")

    print("=== SEEDED BROKEN FILE (sandbox/buggy_calc.py) ===")
    print(broken_code)

    profile = WorkerProfile(
        model="ibm/granite4.1:3b",
        context_limit="limited",
        strengths=["fixing syntax errors", "fixing arithmetic logic bugs"],
        avoid=["rewriting unrelated code"],
        execution_mode="one_session_per_task",
    )

    goal = """
    Fix the existing `buggy_calc.py` file in the workspace root so that:
    1. The syntax error in `def average(a, b):` is fixed.
    2. The arithmetic bug in `average(a, b)` is fixed so it returns `(a + b) / 2`.
    3. Keep the assertion `assert result == 15.0` and print `Average: 15.0`.
    """

    history = run_project(
        goal=goal,
        architecture="Single-file bug repair in existing `buggy_calc.py`.",
        worker_profile=profile,
        max_cycles=2,
    )

    final_cycle = history[-1]
    print("\n=== FINAL CYCLE RESULT ===")
    print(final_cycle.model_dump_json(indent=2))

    print("\n=== REPAIRED FILE CONTENT ===")
    print(TARGET_FILE.read_text(encoding="utf-8"))

    completed = subprocess.run(
        [sys.executable, str(TARGET_FILE)],
        cwd=WORKSPACE_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    if completed.returncode != 0:
        raise AssertionError(f"Repaired buggy_calc.py failed to run:\n{completed.stderr}")

    output = completed.stdout.strip()
    if output != "Average: 15.0":
        raise AssertionError(f"Unexpected output: {output!r} (expected 'Average: 15.0')")

    if final_cycle.outcome != "overall_success":
        raise AssertionError(f"Expected 'overall_success', got {final_cycle.outcome!r}")

    print("\n=== ALPHA04 BUG-RECOVERY TEST PASSED ===")
    print(f"Output: {output}")


if __name__ == "__main__":
    main()

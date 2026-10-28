from pathlib import Path
import subprocess
import sys

from src.orchestration.graph import WORKER_WORKSPACE, run_project
from src.orchestration.ledger import format_ledger_for_prompt
from src.orchestration.schemas import WorkerProfile


ANALYTICS_FILE = WORKER_WORKSPACE / "analytics.py"
MAIN_FILE = WORKER_WORKSPACE / "main.py"
STAGING_DIR = WORKER_WORKSPACE / ".staging"


def main() -> None:
    WORKER_WORKSPACE.mkdir(parents=True, exist_ok=True)
    for py_file in WORKER_WORKSPACE.rglob("*.py"):
        py_file.unlink(missing_ok=True)

    profile = WorkerProfile(
        model="ibm/granite4.1:3b",
        context_limit="limited",
        strengths=["writing a single isolated Python function at a time"],
        avoid=["writing large multi-function modules in a single task"],
        execution_mode="one_function_per_task_staged",
    )

    goal = """
    Create a two-file Python score analytics program in the workspace root:

    1. `analytics.py`:
       Implement three standalone functions in `analytics.py`:
       - `parse_scores(raw: str) -> list[int]`:
         Splits `raw` by comma `,` and returns a list of integers (using `int(x.strip())` for each non-empty piece).
       - `filter_passing(scores: list[int]) -> list[int]`:
         Returns a list of scores from `scores` that are greater than or equal to 60 (`s >= 60`).
       - `summarize_scores(raw: str) -> str`:
         Calls `parse_scores(raw)`, passes that list to `filter_passing(...)`, and returns the formatted string:
         `f"Passing count: {len(passing)}, Total: {sum(passing)}"`
       Do not print anything at import time in `analytics.py`.

    2. `main.py`:
       Import `summarize_scores` from `analytics`.
       Call `summarize_scores('45, 70, 85, 50, 95')`, store the result, assert that it equals
       'Passing count: 3, Total: 250', and print the exact result:
       Passing count: 3, Total: 250
    """

    architecture = """
    - `analytics.py` contains 3 distinct functions (`parse_scores`, `filter_passing`, `summarize_scores`) and MUST be decomposed into separate function-level tasks for the Worker and assembled by the Foreman.
    - `main.py` is the entry-point script importing `summarize_scores` from `analytics`.
    """

    history = run_project(
        goal=goal,
        architecture=architecture,
        worker_profile=profile,
        max_cycles=2,
    )

    final_cycle = history[-1]

    print("\n==========================================")
    print("=== EXECUTION LEDGER SUMMARY ===")
    print("==========================================")
    print(format_ledger_for_prompt(final_cycle.ledger))

    # 1. Verify Foreman split analytics.py into multiple atomic tasks
    initial_batch = final_cycle.task_batches[0]
    analytics_tasks = [
        t for t in initial_batch.tasks if "analytics.py" in t.relevant_files
    ]
    print(f"\nTasks targeting analytics.py: {[t.task_id for t in analytics_tasks]}")
    if len(analytics_tasks) < 2:
        raise AssertionError(
            f"Expected Foreman to atomize analytics.py into >=2 function tasks, got {len(analytics_tasks)}"
        )

    # 2. Verify staged files were created in sandbox/.staging/
    staged_files = sorted(STAGING_DIR.glob("*__analytics.py"))
    print(f"Staged snippet files in sandbox/.staging/: {[f.name for f in staged_files]}")
    if len(staged_files) < 2:
        raise AssertionError("Expected staged snippet files in sandbox/.staging/")

    # 3. Verify assembled analytics.py and main.py contents
    print(f"\n=== Assembled {ANALYTICS_FILE.name} ===")
    analytics_content = ANALYTICS_FILE.read_text(encoding="utf-8")
    print(analytics_content)

    for fn in ("def parse_scores(", "def filter_passing(", "def summarize_scores("):
        if fn not in analytics_content:
            raise AssertionError(f"Assembled analytics.py is missing {fn}")

    print(f"=== Final {MAIN_FILE.name} ===")
    print(MAIN_FILE.read_text(encoding="utf-8"))

    # 4. Verify runtime output
    completed = subprocess.run(
        [sys.executable, str(MAIN_FILE)],
        cwd=WORKER_WORKSPACE,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise AssertionError(f"main.py execution failed:\n{completed.stderr}")

    output = completed.stdout.strip()
    if output != "Passing count: 3, Total: 250":
        raise AssertionError(
            f"Unexpected output: {output!r} (expected 'Passing count: 3, Total: 250')"
        )

    if final_cycle.outcome != "overall_success":
        raise AssertionError(f"Expected 'overall_success', got {final_cycle.outcome!r}")

    print("\n=== ALPHA05 ATOMIZATION, STAGING, ASSEMBLY & LEDGER TEST PASSED ===")
    print(f"Atomic tasks for analytics.py: {len(analytics_tasks)}")
    print(f"Ledger entries recorded: {len(final_cycle.ledger)}")
    print(f"Output: {output}")


if __name__ == "__main__":
    main()

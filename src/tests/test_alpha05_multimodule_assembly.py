from pathlib import Path
import subprocess
import sys

from src.orchestration.graph import WORKER_WORKSPACE, run_project
from src.orchestration.ledger import format_ledger_for_prompt
from src.orchestration.schemas import WorkerProfile


CLEANER_FILE = WORKER_WORKSPACE / "cleaner.py"
AGGREGATOR_FILE = WORKER_WORKSPACE / "aggregator.py"
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
    Create a three-file Python expense processing program in the workspace root:

    1. `cleaner.py`:
       Implement two standalone functions in `cleaner.py`:
       - `normalize_name(name: str) -> str`:
         Returns `name.strip().title()`.
       - `parse_amount(raw: str) -> int`:
         Returns `int(raw.strip().lstrip('$'))`.
       Do not print anything at import time in `cleaner.py`.

    2. `aggregator.py`:
       Import `normalize_name` and `parse_amount` from `cleaner`.
       Implement two standalone functions in `aggregator.py`:
       - `parse_expense(line: str) -> tuple[str, int]`:
         Splits `line` on `':'` into two parts (`left, right = line.split(':')`) and returns
         `(normalize_name(left), parse_amount(right))`.
       - `summarize_expenses(lines: list[str]) -> str`:
         Calls `parse_expense(line)` for each line in `lines` to build a list `items`, and returns:
         `f"Count: {len(items)}, Total: ${sum(amt for _, amt in items)}"`
       Do not print anything at import time in `aggregator.py`.

    3. `main.py`:
       Import `summarize_expenses` from `aggregator`.
       Call `summarize_expenses([" alice : $40 ", "bob : $60", " CHARLIE : $50 "])`,
       store the result, assert that it equals `'Count: 3, Total: $150'`, and print the exact result:
       Count: 3, Total: $150
    """

    architecture = """
    - `cleaner.py` has 2 standalone functions (`normalize_name`, `parse_amount`) -> decompose into 2 function-level tasks.
    - `aggregator.py` has 2 standalone functions (`parse_expense`, `summarize_expenses`) -> decompose into 2 function-level tasks that depend on `cleaner.py` tasks.
    - `main.py` is the entry-point script -> 1 task depending on `aggregator.py` tasks.
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

    # 1. Verify both cleaner.py and aggregator.py were atomized into multiple tasks
    initial_batch = final_cycle.task_batches[0]
    cleaner_tasks = [t.task_id for t in initial_batch.tasks if "cleaner.py" in t.relevant_files]
    aggregator_tasks = [t.task_id for t in initial_batch.tasks if "aggregator.py" in t.relevant_files]

    print(f"\nTasks for cleaner.py: {cleaner_tasks}")
    print(f"Tasks for aggregator.py: {aggregator_tasks}")

    assert len(cleaner_tasks) >= 2, f"Expected >=2 tasks for cleaner.py, got {cleaner_tasks}"
    assert len(aggregator_tasks) >= 2, f"Expected >=2 tasks for aggregator.py, got {aggregator_tasks}"

    # 2. Verify staged snippet files exist for both modules
    staged_cleaner = sorted(f.name for f in STAGING_DIR.glob("*__cleaner.py"))
    staged_aggregator = sorted(f.name for f in STAGING_DIR.glob("*__aggregator.py"))
    print(f"Staged cleaner snippets: {staged_cleaner}")
    print(f"Staged aggregator snippets: {staged_aggregator}")

    assert len(staged_cleaner) >= 2, "Expected >=2 staged files for cleaner.py"
    assert len(staged_aggregator) >= 2, "Expected >=2 staged files for aggregator.py"

    # 3. Inspect assembled files
    print(f"\n=== Assembled {CLEANER_FILE.name} ===")
    cleaner_code = CLEANER_FILE.read_text(encoding="utf-8")
    print(cleaner_code)

    print(f"=== Assembled {AGGREGATOR_FILE.name} ===")
    aggregator_code = AGGREGATOR_FILE.read_text(encoding="utf-8")
    print(aggregator_code)

    print(f"=== Final {MAIN_FILE.name} ===")
    print(MAIN_FILE.read_text(encoding="utf-8"))

    for fn in ("def normalize_name(", "def parse_amount("):
        assert fn in cleaner_code, f"cleaner.py missing {fn}"

    for fn in ("def parse_expense(", "def summarize_expenses("):
        assert fn in aggregator_code, f"aggregator.py missing {fn}"

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
    if output != "Count: 3, Total: $150":
        raise AssertionError(
            f"Unexpected output: {output!r} (expected 'Count: 3, Total: $150')"
        )

    assert final_cycle.outcome == "overall_success", f"Expected overall_success, got {final_cycle.outcome}"

    print("\n=== ALPHA05 MULTI-MODULE + MULTI-FUNCTION ASSEMBLY TEST PASSED ===")
    print(f"Total tasks executed: {len(initial_batch.tasks)}")
    print(f"Ledger steps recorded: {len(final_cycle.ledger)}")
    print(f"Output: {output}")


if __name__ == "__main__":
    main()

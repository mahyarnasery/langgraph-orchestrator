from pathlib import Path
import subprocess
import sys

from src.orchestration.graph import WORKER_WORKSPACE, run_project
from src.orchestration.schemas import WorkerProfile


def main() -> None:
    WORKER_WORKSPACE.mkdir(parents=True, exist_ok=True)
    for py_file in WORKER_WORKSPACE.rglob("*.py"):
        py_file.unlink(missing_ok=True)

    profile = WorkerProfile(
        model="ibm/granite4.1:3b",
        context_limit="limited",
        strengths=["localized single-file edits", "writing clean classes and functions"],
        avoid=["architecture redesign", "interactive input() prompts"],
        execution_mode="one_session_per_task",
    )

    goal = """
    Create a three-file Python inventory system in the workspace root:
    1. models.py:
       Define a class Item with __init__(self, name, price, quantity) storing those three attributes, and a method subtotal(self) returning price * quantity.
    2. inventory.py:
       Import Item from models. Define a class Inventory with __init__(self) initializing an empty list self.items, a method add_item(self, item) that appends item to self.items, and a method total_value(self) returning the sum of item.subtotal() for all items in self.items.
    3. main.py:
       Import Item from models and Inventory from inventory. Create an Inventory instance, add Item('Laptop', 1000, 2) and Item('Mouse', 50, 3), verify with assert that total_value() == 2150, and print the exact line:
       Total Inventory Value: 2150
    """

    history = run_project(
        goal=goal,
        architecture="Three-module Python package in sandbox/: models.py, inventory.py, and main.py.",
        worker_profile=profile,
        max_cycles=2,
    )

    final = history[-1]
    assert final.outcome == "overall_success", f"Expected overall_success, got {final.outcome}"

    for fname in ("models.py", "inventory.py", "main.py"):
        assert (WORKER_WORKSPACE / fname).is_file(), f"Missing {fname}"

    completed = subprocess.run(
        [sys.executable, str(WORKER_WORKSPACE / "main.py")],
        cwd=WORKER_WORKSPACE,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, f"main.py failed: {completed.stderr}"
    assert completed.stdout.strip() == "Total Inventory Value: 2150"

    print("\n=== ALPHA04 3-MODULE OOP TEST PASSED ===")


if __name__ == "__main__":
    main()

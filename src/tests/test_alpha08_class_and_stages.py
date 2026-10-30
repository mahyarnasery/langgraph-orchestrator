import os
import shutil
import subprocess
import sys

from src.orchestration.checkpoint import load_checkpoint
import src.orchestration.graph as graph_mod
from src.orchestration.graph import (
    WORKER_WORKSPACE,
    foreman_self_repair_node,
    route_after_foreman_review,
    run_project,
    run_staged_project,
)
from src.orchestration.ledger import (
    extract_all_ast_symbol_names,
    extract_python_signatures,
    format_ledger_for_prompt,
    merge_staged_snippets_ast,
    record_ledger_entry,
    upsert_staged_snippet,
)
from src.orchestration.schemas import (
    DirectRepairPatch,
    ForemanReview,
    StageGoal,
    TaskResult,
    ValidationResult,
    WorkerProfile,
)


STAGING_DIR = WORKER_WORKSPACE / ".staging"


def clean_workspace() -> None:
    WORKER_WORKSPACE.mkdir(parents=True, exist_ok=True)
    STAGING_DIR.mkdir(parents=True, exist_ok=True)
    for py_file in WORKER_WORKSPACE.rglob("*.py"):
        py_file.unlink(missing_ok=True)
    for child in sorted(WORKER_WORKSPACE.iterdir()):
        if child.is_dir() and child.name != ".git":
            shutil.rmtree(child, ignore_errors=True)


def test_part1_ast_class_method_merging_and_stagnation_guard() -> None:
    """
    Part 1: Verify:
    A) Deterministic AST class method merging, stub guard, and method-level upsert.
    B) Stagnation Circuit Breaker:
       - Zero-diff `foreman_self_repair_node` immediately exhausts `foreman_repair_attempts`
         so identical patches cannot loop.
       - `route_after_foreman_review` trips `[STAGNATION CIRCUIT BREAKER]` when identical
         validator errors repeat without progress.
    """
    print("\n================================================================")
    print("=== PART 1: AST CLASS MERGING & STAGNATION CIRCUIT BREAKER ===")
    print("================================================================")
    clean_workspace()

    snippet_t1 = (
        "class Wallet:\n"
        "    def __init__(self, owner: str, balance: int = 0) -> None:\n"
        "        self.owner = owner\n"
        "        self.balance = balance\n\n"
        "    def deposit(self, amount: int) -> int:\n"
        "        pass\n"
    )
    snippet_t2_buggy = (
        "class Wallet:\n"
        "    def __init__(self, owner: str, balance: int = 0) -> None:\n"
        "        ...\n\n"
        "    def deposit(self, amount: int) -> int:\n"
        "        self.balance += amount + 999\n"
        "        return self.balance\n"
    )
    snippet_t3_bare = (
        "def summary(self) -> str:\n"
        "    return f'{self.owner}: ${self.balance}'\n"
    )

    snippets: dict[str, str] = {}
    snippets = upsert_staged_snippet(snippets, "t1", snippet_t1, target_symbol="Wallet.__init__")
    snippets = upsert_staged_snippet(snippets, "t2", snippet_t2_buggy, target_symbol="Wallet.deposit")
    snippets = upsert_staged_snippet(snippets, "t3", snippet_t3_bare, target_symbol="Wallet.summary")

    assert list(snippets.keys()) == ["t1", "t2", "t3"]

    snippet_t2_fixed = (
        "class Wallet:\n"
        "    def deposit(self, amount: int) -> int:\n"
        "        self.balance += amount\n"
        "        return self.balance\n"
    )
    snippets = upsert_staged_snippet(
        snippets, "t2_repaired", snippet_t2_fixed, target_symbol="Wallet.deposit"
    )
    assert "t2" not in snippets and "t2_repaired" in snippets

    merged_code = merge_staged_snippets_ast(
        target_filename="wallet.py",
        base_source="",
        ordered_snippets=list(snippets.items()),
    )
    assert merged_code.count("class Wallet:") == 1
    symbols = extract_all_ast_symbol_names(merged_code)
    for expected_sym in ("Wallet", "Wallet.__init__", "Wallet.deposit", "Wallet.summary"):
        assert expected_sym in symbols

    sigs = extract_python_signatures(merged_code)
    assert any("self.owner" in s and "self.balance" in s for s in sigs)

    # --- Test B: Zero-Diff Self-Repair Guard & Error Stagnation Circuit Breaker ---
    dummy_file = WORKER_WORKSPACE / "dummy.py"
    dummy_file.write_text("x = 1\n", encoding="utf-8")

    profile = WorkerProfile(
        model="ibm/granite4.1:3b",
        context_limit="limited",
        strengths=[],
        avoid=[],
        execution_mode="aider",
    )

    noop_review = ForemanReview(
        decision="repair",
        accepted=False,
        direct_repairs=[
            DirectRepairPatch(path="dummy.py", action="overwrite", content="x = 1\n")
        ],
        summary="Applying identical content",
    )

    mock_state = {
        "cycle_id": 1,
        "project_goal": "test",
        "architecture_context": "test",
        "worker_profile": profile,
        "previous_cycle": None,
        "plan": None,
        "task_batches": [],
        "task_results": [],
        "ledger": [],
        "staged_files": {},
        "file_snapshots": {},
        "worker_attempts": {},
        "foreman_repair_attempts": 0,
        "rebuild_attempts": 1,
        "validation_result": ValidationResult(
            passed=False,
            checks=[],
            errors=["assertion failed (t1): assert x == 2 [actual=1, expected=2]"],
            files_checked=["dummy.py"],
            changed_files=["dummy.py"],
            failed_task_ids=["t1"],
        ),
        "foreman_review": noop_review,
        "planner_review": None,
        "rework_mode": "none",
        "phase": "foreman_review_repair",
    }

    repair_updates = foreman_self_repair_node(mock_state)  # type: ignore[arg-type]
    print(f"foreman_repair_attempts after zero-diff patch: {repair_updates['foreman_repair_attempts']}")
    assert repair_updates["foreman_repair_attempts"] == graph_mod.MAX_FOREMAN_REPAIRS, (
        "Expected zero-diff self-repair to exhaust foreman_repair_attempts immediately!"
    )

    # Simulate 2 consecutive identical validator failures + no_change result -> must trip circuit breaker
    ledger = []
    for _ in range(2):
        ledger = record_ledger_entry(
            ledger,
            cycle_id=1,
            layer="validator",
            action="validate_batch",
            target="dummy.py",
            status="failure",
            detail="Failed (t1): assertion failed (t1): assert x == 2 [actual=1, expected=2]",
        )
    mock_state["ledger"] = ledger
    mock_state["task_results"] = repair_updates["task_results"]
    mock_state["foreman_repair_attempts"] = repair_updates["foreman_repair_attempts"]

    route_decision = route_after_foreman_review(mock_state)  # type: ignore[arg-type]
    print(f"Router decision on stagnant loop: {route_decision}")
    assert route_decision == "accepted", (
        f"Expected Stagnation Circuit Breaker to break loop ('accepted'), got {route_decision}"
    )
    print("=== PART 1 PASSED ===")


def test_part2_end_to_end_class_method_atomization() -> None:
    """
    Part 2: Full end-to-end orchestration cycle where Foreman atomizes a multi-method
    class (`Account` in `bank.py` with `__init__`, `deposit`, `describe`) into separate
    method-level tasks, stages each method in `.staging/`, merges them into `class Account:`,
    verifies per-method unit assertions, and executes `main.py`.
    """
    print("\n================================================================")
    print("=== PART 2: END-TO-END CLASS METHOD ATOMIZATION & ASSEMBLY ===")
    print("================================================================")
    clean_workspace()

    profile = WorkerProfile(
        model="ibm/granite4.1:3b",
        context_limit="limited",
        strengths=["writing a single isolated Python function or class method at a time"],
        avoid=["writing large multi-method classes in a single task"],
        execution_mode="one_method_per_task_staged",
    )

    goal = """
    Create a two-file Python bank account program in the workspace root:

    1. `bank.py`:
       Define a class `Account` in `bank.py` with three methods:
       - `__init__(self, owner: str, balance: int = 0) -> None`:
         Stores `self.owner = owner` and `self.balance = balance`.
       - `deposit(self, amount: int) -> int`:
         Adds `amount` to `self.balance` (`self.balance += amount`) and returns `self.balance`.
       - `describe(self) -> str`:
         Returns the formatted string `f"Account[{self.owner}]: ${self.balance}"`.
       Do not print anything at import time in `bank.py`.

    2. `main.py`:
       Import `Account` from `bank`.
       Create `acc = Account("Alice", 100)`, call `acc.deposit(75)`,
       store `res = acc.describe()`, assert that `res == "Account[Alice]: $175"`,
       and print the exact result:
       Account[Alice]: $175
    """

    architecture = """
    - `bank.py` defines `class Account` with 3 methods (`__init__`, `deposit`, `describe`) -> MUST be decomposed into 3 separate method-level tasks (`target_symbol="Account.__init__"`, `"Account.deposit"`, `"Account.describe"`).
    - `main.py` is the entry-point script importing `Account` from `bank`.
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

    initial_batch = final_cycle.task_batches[0]
    bank_tasks = [t for t in initial_batch.tasks if "bank.py" in t.relevant_files]
    print(f"\nTasks targeting bank.py: {[(t.task_id, t.target_symbol) for t in bank_tasks]}")
    assert len(bank_tasks) >= 2, f"Expected >=2 method tasks for bank.py, got {len(bank_tasks)}"

    staged_bank_files = sorted(f.name for f in STAGING_DIR.glob("*__bank.py"))
    print(f"Staged method files for bank.py: {staged_bank_files}")
    assert len(staged_bank_files) >= 2, "Expected >=2 staged snippet files for bank.py"

    bank_code = (WORKER_WORKSPACE / "bank.py").read_text(encoding="utf-8")
    print("\n=== Assembled bank.py ===")
    print(bank_code)

    bank_symbols = extract_all_ast_symbol_names(bank_code)
    for sym in ("Account", "Account.__init__", "Account.deposit", "Account.describe"):
        assert sym in bank_symbols, f"Assembled bank.py missing {sym} (found {bank_symbols})"

    completed = subprocess.run(
        [sys.executable, str(WORKER_WORKSPACE / "main.py")],
        cwd=WORKER_WORKSPACE,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, f"main.py failed:\n{completed.stderr}"
    assert completed.stdout.strip() == "Account[Alice]: $175"
    assert final_cycle.outcome == "overall_success"
    print("=== PART 2 PASSED ===")


def test_part3_multistage_blueprint_interrupt_and_resume() -> None:
    """
    Part 3: Multi-Stage Blueprint Decomposition with mid-stage interrupt & resume:
    - Stage 1: Build `models/product.py` with multi-method `class Product`
      (`__init__`, `line_total`, `label`).
    - Stage 2: Build `services/invoice.py` (2 functions: `compute_grand_total`,
      `render_invoice` importing `Product` from `models.product`) and `main.py`.
    - Interrupt Stage 2 after 1 Worker task (`ORCHESTRATOR_STOP_AFTER_TASKS=1`), verify
      Stage 1 is locked in as completed in `sandbox/.checkpoint/state.json`, and resume
      with `run_staged_project(..., resume=True)` to completion.
    """
    print("\n================================================================")
    print("=== PART 3: MULTI-STAGE DECOMPOSITION, INTERRUPT & RESUME ===")
    print("================================================================")
    clean_workspace()

    profile = WorkerProfile(
        model="ibm/granite4.1:3b",
        context_limit="limited",
        strengths=["writing a single isolated Python function or class method at a time"],
        avoid=["writing large multi-file modules in a single task"],
        execution_mode="staged_multi_milestone",
    )

    full_goal = "Multi-package Python invoice system with models/product.py, services/invoice.py, and main.py."

    stages = [
        StageGoal(
            stage_id=1,
            title="Core Product Domain Model",
            objective=(
                "Create `models/product.py` defining `class Product` with three methods:\n"
                "- `__init__(self, name: str, price: int, qty: int) -> None`: sets `self.name = name`, `self.price = price`, `self.qty = qty`.\n"
                "- `line_total(self) -> int`: returns `self.price * self.qty`.\n"
                "- `label(self) -> str`: returns `f\"{self.name} (x{self.qty})\"`.\n"
                "Split `class Product` into 3 method-level tasks (`Product.__init__`, `Product.line_total`, `Product.label`)."
            ),
            target_files=["models/product.py"],
        ),
        StageGoal(
            stage_id=2,
            title="Invoice Service & Main Entry Script",
            objective=(
                "Keep existing `models/product.py` intact and create `services/invoice.py` and `main.py`:\n"
                "1. In `services/invoice.py`, import `Product` from `models.product` and implement two standalone functions:\n"
                "   - `compute_grand_total(items: list[Product]) -> int`: returns `sum(item.line_total() for item in items)`.\n"
                "   - `render_invoice(items: list[Product]) -> str`: calls `compute_grand_total(items)` and returns `f\"Items: {len(items)}, Grand Total: ${compute_grand_total(items)}\"`.\n"
                "2. In `main.py`, import `Product` from `models.product` and `render_invoice` from `services.invoice`, "
                "create `items = [Product('Book', 15, 2), Product('Pen', 5, 4)]`, assert `render_invoice(items) == 'Items: 2, Grand Total: $50'`, "
                "and print the exact line:\n"
                "Items: 2, Grand Total: $50"
            ),
            target_files=["services/invoice.py", "main.py"],
        ),
    ]

    architecture = (
        "Two-stage multi-package Python project: `models/product.py` (Stage 1), "
        "`services/invoice.py` and `main.py` (Stage 2)."
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
        )
    except KeyboardInterrupt as exc:
        stage2_interrupted = True
        print(f"\n[TEST] Caught expected mid-Stage-2 interrupt: {exc}")
    finally:
        graph_mod.run_cycle = orig_run_cycle

    assert stage2_interrupted, "Expected KeyboardInterrupt during Stage 2!"

    ckpt_mid = load_checkpoint()
    assert ckpt_mid is not None
    print(
        f"[TEST] Checkpoint state at interrupt: current_stage_index={ckpt_mid.current_stage_index}, "
        f"completed_stage_ids={ckpt_mid.completed_stage_ids}, "
        f"stage2_completed_tasks={[r.task_id for r in ckpt_mid.task_results]}"
    )
    assert ckpt_mid.completed_stage_ids == [1], f"Expected Stage 1 completed, got {ckpt_mid.completed_stage_ids}"
    assert ckpt_mid.current_stage_index == 1, f"Expected current_stage_index == 1, got {ckpt_mid.current_stage_index}"
    assert (WORKER_WORKSPACE / "models" / "product.py").is_file(), "Stage 1 models/product.py must exist on disk"

    print("\n[TEST] Resuming multi-stage project with resume=True...")
    history = run_staged_project(
        goal=full_goal,
        architecture=architecture,
        worker_profile=profile,
        stages=stages,
        max_cycles_per_stage=2,
        resume=True,
    )

    final_cycle = history[-1]
    ckpt_final = load_checkpoint()
    assert ckpt_final is not None
    print(f"[TEST] Final completed_stage_ids: {ckpt_final.completed_stage_ids}")
    print(f"[TEST] Stage 2 worker_attempts: {ckpt_final.worker_attempts}")

    assert ckpt_final.completed_stage_ids == [1, 2], (
        f"Expected stages [1, 2] completed, got {ckpt_final.completed_stage_ids}"
    )

    first_stage2_tid = ckpt_mid.task_results[0].task_id
    assert ckpt_final.worker_attempts.get(first_stage2_tid) == 1, (
        f"Expected Stage 2 task {first_stage2_tid} to run only once across interrupt+resume, got {ckpt_final.worker_attempts}"
    )

    completed = subprocess.run(
        [sys.executable, str(WORKER_WORKSPACE / "main.py")],
        cwd=WORKER_WORKSPACE,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, f"main.py failed:\n{completed.stderr}"
    output = completed.stdout.strip()
    assert output == "Items: 2, Grand Total: $50", f"Unexpected output: {output!r}"
    assert final_cycle.outcome == "overall_success"

    print("\n=== ALPHA08 ALL CLASS METHOD ATOMIZATION, STAGNATION GUARD & MULTI-STAGE TESTS PASSED ===")
    print(f"Final Output: {output}")


def main() -> None:
    test_part1_ast_class_method_merging_and_stagnation_guard()
    test_part2_end_to_end_class_method_atomization()
    test_part3_multistage_blueprint_interrupt_and_resume()


if __name__ == "__main__":
    main()

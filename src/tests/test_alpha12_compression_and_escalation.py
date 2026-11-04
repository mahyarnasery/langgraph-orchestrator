from pathlib import Path
import shutil
import subprocess
import sys

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from langgraph.graph import END, StateGraph
import src.orchestration.graph as graph_mod
from src.harnesses.aider import build_inline_dependency_context
from src.orchestration.graph import WORKER_WORKSPACE, _gather_repository_context
from src.orchestration.ledger import format_ledger_for_prompt, generate_pyi_stub
from src.orchestration.schemas import (
    Plan,
    Task,
    TaskBatch,
    TaskResult,
    ValidationResult,
    WorkerProfile,
)
from src.orchestration.state import WorkflowState
from src.orchestration.validator import emit_pytest_suite


STAGING_DIR = WORKER_WORKSPACE / ".staging"


def clean_workspace() -> None:
    WORKER_WORKSPACE.mkdir(parents=True, exist_ok=True)
    for child in sorted(WORKER_WORKSPACE.iterdir()):
        if child.name == ".git":
            continue
        if child.is_dir():
            shutil.rmtree(child, ignore_errors=True)
        else:
            child.unlink(missing_ok=True)


def test_part1_pyi_stub_projection_and_failure_scoped_context() -> None:
    """
    Part 1: Verify:
    1. `generate_pyi_stub()` strips method docstrings, comments, and multi-line bodies
       down to `...` while preserving class attributes (`self.x`), signatures, and return types.
    2. `build_inline_dependency_context()` embeds `.pyi` stubs and `.json` ground-truth literals.
    3. `_gather_repository_context()` omits full source dumps for passing files when another
       file in the batch failed validation.
    """
    print("\n==================================================================")
    print("=== PART 1: .PYI STUB PROJECTION & FAILURE-SCOPED CONTEXT ===")
    print("==================================================================")
    clean_workspace()

    verbose_domain_source = """'''Verbose module docstring that should not bloat worker prompts.'''

MAX_SPILLWAY_GATES = 4

class ReservoirModel:
    '''Detailed class docstring with many lines of prose explanation.'''

    def __init__(self, station_id: str, capacity_mcm: float = 500.0) -> None:
        # Store primary attributes
        self.station_id = station_id.strip().upper()
        self.capacity_mcm = float(capacity_mcm)
        self.history = []

    def compute_storage_ratio(self, current_mcm: float, reserve_mcm: float = 50.0) -> float:
        '''Calculate usable storage ratio after subtracting dead storage reserve.'''
        usable = max(0.0, current_mcm - reserve_mcm)
        denom = max(1.0, self.capacity_mcm - reserve_mcm)
        ratio = usable / denom
        return round(min(1.0, ratio), 4)
"""
    stub = generate_pyi_stub(verbose_domain_source)
    print("Generated .pyi stub:\n" + stub)

    assert "MAX_SPILLWAY_GATES = 4" in stub
    assert "class ReservoirModel:" in stub
    assert "station_id: Any" in stub and "capacity_mcm: Any" in stub
    assert "def __init__(self, station_id: str, capacity_mcm: float=500.0) -> None: ..." in stub
    assert "def compute_storage_ratio(self, current_mcm: float, reserve_mcm: float=50.0) -> float: ..." in stub
    assert "usable = max(" not in stub, "Expected function body to be stripped from .pyi stub"
    assert len(stub) < len(verbose_domain_source) * 0.55, "Expected >45% character compression"

    # Write domain & config files to workspace and test inline dependency context
    (WORKER_WORKSPACE / "domain").mkdir(parents=True, exist_ok=True)
    (WORKER_WORKSPACE / "config").mkdir(parents=True, exist_ok=True)
    (WORKER_WORKSPACE / "domain" / "reservoir.py").write_text(verbose_domain_source, encoding="utf-8")
    (WORKER_WORKSPACE / "config" / "limits.json").write_text('{"max_head_m": 120.0}\n', encoding="utf-8")
    (WORKER_WORKSPACE / "services").mkdir(parents=True, exist_ok=True)
    (WORKER_WORKSPACE / "services" / "calc.py").write_text("def broken_fn(x):\n    return x + 999\n", encoding="utf-8")

    inline_ctx = build_inline_dependency_context(
        ["config/limits.json", "domain/reservoir.py"],
        WORKER_WORKSPACE,
    )
    print("\nInline Dependency Context:\n" + inline_ctx)
    assert "def compute_storage_ratio(" in inline_ctx
    assert '{"max_head_m": 120.0}' in inline_ctx

    # Verify failure-scoped _gather_repository_context
    batch = TaskBatch(
        batch_id="b_test",
        objective="Test failure scoping",
        tasks=[
            Task(task_id="t1", objective="domain", target_symbol="ReservoirModel", scope="domain/reservoir.py", relevant_files=["domain/reservoir.py"], acceptance_criteria=[]),
            Task(task_id="t2", objective="service", target_symbol="broken_fn", scope="services/calc.py", relevant_files=["services/calc.py"], acceptance_criteria=[]),
        ],
        relevant_files=["domain/reservoir.py", "services/calc.py"],
        execution_order=["t1", "t2"],
    )
    results = [
        TaskResult(task_id="t1", status="completed", summary="ok", files_changed=["domain/reservoir.py"]),
        TaskResult(task_id="t2", status="completed", summary="ok", files_changed=["services/calc.py"]),
    ]
    val_fail = ValidationResult(
        passed=False,
        checks=[],
        errors=["assertion failed (t2 [services/calc.py::broken_fn]): assert broken_fn(1) == 2"],
        files_checked=["domain/reservoir.py", "services/calc.py"],
        changed_files=["domain/reservoir.py", "services/calc.py"],
        failed_task_ids=["t2"],
    )
    scoped_repo_ctx = _gather_repository_context(batch, results, validation_result=val_fail)
    print("\nFailure-Scoped Foreman Review Context:\n" + scoped_repo_ctx)
    assert "FILE: domain/reservoir.py — [PASSED ALL CHECKS;" in scoped_repo_ctx
    assert "usable = max(" not in scoped_repo_ctx, "Passing file body should be omitted!"
    assert "return x + 999" in scoped_repo_ctx, "Failed file body must be included!"
    print("=== PART 1 PASSED ===")


def test_part2_3b_to_8b_escalation_and_main_pytest_emission() -> None:
    """
    Part 2: Seed a 3-function staged module (`hydro_math.py`) + `main.py` (with `def main():`)
    where `t2` (`compute_spillway_discharge`) has a deliberate bug (`* 999.0` instead of
    `round(2.0 * width_m * (h ** 1.5), 2)`).
    With `MAX_FOREMAN_REPAIRS = 0`, verify that:
    1. Validator catches the assertion failure on `t2`.
    2. Foreman issues a surgical corrective batch for `t2`.
    3. `worker_node` escalates the rebuild of `t2` directly to the 8B Foreman
       (`ForemanAgent.implement_atomic_task`), which rewrites `.staging/t2__hydro_math.py`.
    4. AST assembly merges `t1`, `t2`, and `t3` into `hydro_math.py`, validation passes,
       and `emit_pytest_suite()` exports unit tests for both the library functions AND `def main():`.
    """
    print("\n==================================================================")
    print("=== PART 2: 3B -> 8B ATOMIC ESCALATION & DEF MAIN() PYTEST ===")
    print("==================================================================")
    clean_workspace()
    STAGING_DIR.mkdir(parents=True, exist_ok=True)

    s1 = (
        "def effective_overtop(head_m: float, crest_m: float) -> float:\n"
        "    return max(0.0, float(head_m) - float(crest_m))\n"
    )
    s2_buggy = (
        "def compute_spillway_discharge(head_m: float, crest_m: float, width_m: float = 10.0) -> float:\n"
        "    return float(head_m) * 999.0\n"
    )
    s3 = (
        "def format_discharge_alert(station: str, head_m: float, crest_m: float) -> str:\n"
        "    q = compute_spillway_discharge(head_m, crest_m)\n"
        "    return f'{station}: Q={q:.1f} cms'\n"
    )

    (STAGING_DIR / "t1__hydro_math.py").write_text(s1, encoding="utf-8")
    (STAGING_DIR / "t2__hydro_math.py").write_text(s2_buggy, encoding="utf-8")
    (STAGING_DIR / "t3__hydro_math.py").write_text(s3, encoding="utf-8")

    hydro_file = WORKER_WORKSPACE / "hydro_math.py"
    main_file = WORKER_WORKSPACE / "main.py"

    hydro_file.write_text(f"{s1}\n\n{s2_buggy}\n\n{s3}", encoding="utf-8")
    main_file.write_text(
        "from hydro_math import format_discharge_alert\n\n"
        "def main() -> None:\n"
        "    report = format_discharge_alert('Karun-4', 104.0, 100.0)\n"
        "    assert report == 'Karun-4: Q=160.0 cms'\n"
        "    print(report)\n\n"
        "if __name__ == '__main__':\n"
        "    main()\n",
        encoding="utf-8",
    )

    plan = Plan(
        objective="Implement hydro_math.py and main.py.",
        approach="Three staged functions in hydro_math.py and main.py entry script.",
        affected_areas=["hydro_math.py", "main.py"],
        acceptance_criteria=[
            "effective_overtop(head_m, crest_m) returns max(0.0, float(head_m) - float(crest_m)).",
            "compute_spillway_discharge(head_m, crest_m, width_m=10.0) calls h = effective_overtop(head_m, crest_m) and returns round(2.0 * width_m * (h ** 1.5), 2).",
            "format_discharge_alert(station, head_m, crest_m) returns f'{station}: Q={compute_spillway_discharge(head_m, crest_m):.1f} cms'.",
            "main.py prints 'Karun-4: Q=160.0 cms'.",
        ],
    )

    initial_batch = TaskBatch(
        batch_id="batch_hydro_alpha12",
        objective=plan.objective,
        tasks=[
            Task(
                task_id="t1",
                objective="Implement effective_overtop(head_m: float, crest_m: float) -> float returning max(0.0, float(head_m) - float(crest_m))",
                target_symbol="effective_overtop",
                scope="hydro_math.py",
                relevant_files=["hydro_math.py"],
                acceptance_criteria=["def effective_overtop(head_m: float, crest_m: float) -> float:"],
                test_assertions=["assert effective_overtop(104.0, 100.0) == 4.0"],
            ),
            Task(
                task_id="t2",
                objective=(
                    "Implement compute_spillway_discharge(head_m: float, crest_m: float, width_m: float = 10.0) -> float "
                    "calling h = effective_overtop(head_m, crest_m) and returning round(2.0 * width_m * (h ** 1.5), 2)"
                ),
                target_symbol="compute_spillway_discharge",
                scope="hydro_math.py",
                relevant_files=["hydro_math.py"],
                dependencies=["t1"],
                acceptance_criteria=[
                    "def compute_spillway_discharge(head_m: float, crest_m: float, width_m: float = 10.0) -> float:",
                    "Returns round(2.0 * width_m * (effective_overtop(head_m, crest_m) ** 1.5), 2)",
                ],
                test_assertions=[
                    "assert compute_spillway_discharge(104.0, 100.0, 10.0) == 160.0",
                    "assert compute_spillway_discharge(98.0, 100.0, 10.0) == 0.0",
                ],
            ),
            Task(
                task_id="t3",
                objective="Implement format_discharge_alert(station: str, head_m: float, crest_m: float) -> str",
                target_symbol="format_discharge_alert",
                scope="hydro_math.py",
                relevant_files=["hydro_math.py"],
                dependencies=["t1", "t2"],
                acceptance_criteria=["def format_discharge_alert(station: str, head_m: float, crest_m: float) -> str:"],
                test_assertions=["assert format_discharge_alert('Karun-4', 104.0, 100.0) == 'Karun-4: Q=160.0 cms'"],
            ),
            Task(
                task_id="t4",
                objective="Create main.py printing format_discharge_alert('Karun-4', 104.0, 100.0)",
                target_symbol=None,
                scope="main.py",
                relevant_files=["main.py"],
                dependencies=["t3"],
                acceptance_criteria=["main.py prints 'Karun-4: Q=160.0 cms'"],
                test_assertions=[],
            ),
        ],
        relevant_files=["hydro_math.py", "main.py"],
        execution_order=["t1", "t2", "t3", "t4"],
    )

    initial_results = [
        TaskResult(task_id="t1", status="completed", summary="Staged t1", files_changed=["hydro_math.py"], staged_file=".staging/t1__hydro_math.py"),
        TaskResult(task_id="t2", status="completed", summary="Staged t2", files_changed=["hydro_math.py"], staged_file=".staging/t2__hydro_math.py"),
        TaskResult(task_id="t3", status="completed", summary="Staged t3", files_changed=["hydro_math.py"], staged_file=".staging/t3__hydro_math.py"),
        TaskResult(task_id="t4", status="completed", summary="Created main.py", files_changed=["main.py"]),
    ]

    profile = WorkerProfile(
        model="ibm/granite4.1:3b",
        context_limit="compressed",
        strengths=["isolated functions"],
        avoid=["multi-file rewrites"],
        execution_mode="staged_with_8b_escalation",
    )

    orig_max_repairs = graph_mod.MAX_FOREMAN_REPAIRS
    graph_mod.MAX_FOREMAN_REPAIRS = 0

    try:
        rebuild_graph = StateGraph(WorkflowState)
        rebuild_graph.add_node("validator", graph_mod.validator_node)
        rebuild_graph.add_node("foreman_review", graph_mod.foreman_review_node)
        rebuild_graph.add_node("foreman_corrective", graph_mod.foreman_corrective_node)
        rebuild_graph.add_node("worker", graph_mod.worker_node)
        rebuild_graph.add_node("foreman_assemble", graph_mod.foreman_assemble_node)
        rebuild_graph.add_node("planner_review", graph_mod.planner_review_node)

        rebuild_graph.set_entry_point("validator")
        rebuild_graph.add_edge("validator", "foreman_review")
        rebuild_graph.add_conditional_edges(
            "foreman_review",
            graph_mod.route_after_foreman_review,
            {
                "rebuild": "foreman_corrective",
                "accepted": "planner_review",
            },
        )
        rebuild_graph.add_edge("foreman_corrective", "worker")
        rebuild_graph.add_edge("worker", "foreman_assemble")
        rebuild_graph.add_edge("foreman_assemble", "validator")
        rebuild_graph.add_edge("planner_review", END)

        test_app = rebuild_graph.compile()

        final_state = test_app.invoke(
            {
                "cycle_id": 1,
                "project_goal": plan.objective,
                "architecture_context": "Staged function assembly with 8B escalation.",
                "worker_profile": profile,
                "previous_cycle": None,
                "plan": plan,
                "task_batches": [initial_batch],
                "task_results": initial_results,
                "ledger": [],
                "staged_files": {
                    "hydro_math.py": {
                        "t1": s1.strip(),
                        "t2": s2_buggy.strip(),
                        "t3": s3.strip(),
                    }
                },
                "file_snapshots": {"hydro_math.py": "", "main.py": ""},
                "worker_attempts": {"t1": 1, "t2": 1, "t3": 1, "t4": 1},
                "foreman_repair_attempts": 0,
                "rebuild_attempts": 0,
                "validation_result": None,
                "foreman_review": None,
                "planner_review": None,
                "rework_mode": "none",
                "phase": "assembled",
            }
        )
    finally:
        graph_mod.MAX_FOREMAN_REPAIRS = orig_max_repairs

    print("\n=== PART 2 TELEGRAPHIC LEDGER ===")
    print(format_ledger_for_prompt(final_state["ledger"]))

    final_hydro = hydro_file.read_text(encoding="utf-8")
    print("\n=== Re-Assembled hydro_math.py (via 8B Escalation) ===")
    print(final_hydro)

    assert "999" not in final_hydro, "Buggy 999.0 was not replaced by 8B escalation!"
    assert final_state["validation_result"].passed is True
    assert final_state["planner_review"].accepted is True

    # Verify emit_pytest_suite harvests both library assertions AND `def main():`
    suite_path = emit_pytest_suite(initial_batch, suite_rel_path="tests/test_alpha12_suite.py")
    assert suite_path is not None and suite_path.is_file()
    suite_code = suite_path.read_text(encoding="utf-8")
    print("\n=== Emitted tests/test_alpha12_suite.py ===")
    print(suite_code)
    assert "def test_entry_main_main() -> None:" in suite_code, "Expected `def main():` to be harvested into pytest suite!"

    print("\n=== ALPHA12 .PYI STUBS, FAILURE-SCOPED CONTEXT & 8B ESCALATION TEST PASSED ===")


def main() -> None:
    test_part1_pyi_stub_projection_and_failure_scoped_context()
    test_part2_3b_to_8b_escalation_and_main_pytest_emission()


if __name__ == "__main__":
    main()

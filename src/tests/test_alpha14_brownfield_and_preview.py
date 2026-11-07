import json
import shutil
import subprocess
import sys

from src.orchestration.checkpoint import clear_checkpoint, inspect_workspace_baseline
from src.orchestration.graph import (
    WORKER_WORKSPACE,
    format_plan_preview,
    run_project,
    run_staged_project,
)
from src.orchestration.ledger import (
    extract_all_ast_symbol_names,
    extract_symbol_source_ast,
    rank_repository_files_by_impact,
)
from src.orchestration.schemas import (
    FileBlueprint,
    Plan,
    StageGoal,
    SymbolSpec,
    Task,
    TaskBatch,
    TaskResult,
    WorkerProfile,
)
from src.orchestration.validator import validate_batch


def clean_workspace() -> None:
    WORKER_WORKSPACE.mkdir(parents=True, exist_ok=True)
    for fpath in WORKER_WORKSPACE.rglob("*"):
        if fpath.is_file() and ".git" not in fpath.parts:
            fpath.unlink(missing_ok=True)
    for child in sorted(WORKER_WORKSPACE.iterdir()):
        if child.is_dir() and child.name != ".git":
            shutil.rmtree(child, ignore_errors=True)
    clear_checkpoint()


def test_part1_ast_impact_slicer_non_regression_and_preview() -> None:
    """
    Part 1: Deterministic unit verification of:
    1. `extract_symbol_source_ast()` surgical extraction of functions and class methods.
    2. `rank_repository_files_by_impact()` separating focal + 1-hop modules from background modules.
    3. `validate_batch()` Brownfield Non-Regression AST Symbol Guard catching dropped pre-existing symbols.
    4. `format_plan_preview()` rendering the CLI Spec & Blueprint Preview for `--plan-only`.
    """
    print("\n================================================================")
    print("=== PART 1: AST IMPACT SLICER, NON-REGRESSION & CLI PREVIEW ===")
    print("================================================================")
    clean_workspace()

    # 1. Verify surgical AST symbol extraction
    sample_mod = (
        "import math\n\n"
        "class Turbine:\n"
        "    def __init__(self, tid: str, mw: float):\n"
        "        self.tid = tid\n"
        "        self.mw = mw\n\n"
        "    def output_kw(self) -> float:\n"
        "        return self.mw * 1000.0\n"
    )
    extracted = extract_symbol_source_ast(sample_mod, "Turbine.output_kw")
    assert "class Turbine:" in extracted
    assert "def output_kw(self) -> float:" in extracted
    assert "def __init__" not in extracted, f"Expected only output_kw in surgical extract:\n{extracted}"

    # 2. Verify AST Impact Slicer on >5 files
    repo_sources = {
        "config/settings.json": '{"rate": 0.1}',
        "domain/models.py": "class Invoice:\n    def __init__(self, amt: float):\n        self.amt = amt\n",
        "billing/calculator.py": "from domain.models import Invoice\ndef calc_tax(inv: Invoice) -> float:\n    return inv.amt * 0.1\n",
        "billing/formatter.py": "from billing.calculator import calc_tax\ndef fmt(inv) -> str:\n    return str(calc_tax(inv))\n",
        "audit/legacy_logger.py": "def write_audit_log(msg: str) -> str:\n    return msg.upper()\n",
        "utils/math_extras.py": "def cube(x: int) -> int:\n    return x * x * x\n",
    }
    focal, background = rank_repository_files_by_impact(
        "Update `calc_tax` in `billing/calculator.py` to support tax exemptions.",
        repo_sources,
    )
    print(f"Focal + 1-hop files: {sorted(focal)}")
    print(f"Background files:    {sorted(background)}")
    assert "billing/calculator.py" in focal
    assert "domain/models.py" in focal      # 1-hop imported by calculator.py
    assert "billing/formatter.py" in focal  # 1-hop importer of calculator.py
    assert "audit/legacy_logger.py" in background
    assert "utils/math_extras.py" in background

    # 3. Verify Brownfield Non-Regression AST Symbol Guard
    (WORKER_WORKSPACE / "billing").mkdir(parents=True, exist_ok=True)
    (WORKER_WORKSPACE / "billing" / "__init__.py").write_text("", encoding="utf-8")
    calc_file = WORKER_WORKSPACE / "billing" / "calculator.py"
    pre_snapshot = (
        "def calc_subtotal(items: list[float]) -> float:\n"
        "    return sum(items)\n\n\n"
        "def calc_tax(amount: float) -> float:\n"
        "    return round(amount * 0.1, 2)\n"
    )
    # Simulate an accidental overwrite that added `calc_discount` but dropped `calc_subtotal`
    calc_file.write_text(
        "def calc_tax(amount: float) -> float:\n"
        "    return round(amount * 0.1, 2)\n\n\n"
        "def calc_discount(amount: float) -> float:\n"
        "    return round(amount * 0.05, 2)\n",
        encoding="utf-8",
    )
    batch = TaskBatch(
        batch_id="test_non_regression",
        objective="Add calc_discount to billing/calculator.py",
        tasks=[
            Task(
                task_id="t1",
                objective="Implement calc_discount(amount: float) -> float in billing/calculator.py",
                target_symbol="calc_discount",
                scope="billing/calculator.py",
                relevant_files=["billing/calculator.py"],
                acceptance_criteria=["def calc_discount(amount: float) -> float"],
                test_assertions=["assert calc_discount(100.0) == 5.0"],
            )
        ],
        relevant_files=["billing/calculator.py"],
        execution_order=["t1"],
    )
    res_list = [
        TaskResult(task_id="t1", status="completed", summary="Added calc_discount", files_changed=["billing/calculator.py"])
    ]
    val_reg = validate_batch(batch, res_list, file_snapshots={"billing/calculator.py": pre_snapshot})
    print("Non-regression validator errors (expected failure):", val_reg.errors)
    assert not val_reg.passed, "Expected validator to fail when pre-existing symbol calc_subtotal was dropped!"
    assert any("regression in billing/calculator.py" in e and "calc_subtotal" in e for e in val_reg.errors)

    # Restore calc_subtotal and verify non-regression check passes
    calc_file.write_text(
        pre_snapshot + "\n\ndef calc_discount(amount: float) -> float:\n    return round(amount * 0.05, 2)\n",
        encoding="utf-8",
    )
    val_ok = validate_batch(batch, res_list, file_snapshots={"billing/calculator.py": pre_snapshot})
    assert val_ok.passed, f"Expected non-regression validation to pass, got: {val_ok.errors}"
    assert any("non-regression OK: billing/calculator.py" in c for c in val_ok.checks)

    # 4. Verify CLI Spec & Blueprint Preview formatting
    preview_text = format_plan_preview(
        Plan(
            objective="Extend billing calculator",
            approach="Add calc_discount via staged AST merge",
            affected_areas=["billing/calculator.py"],
            file_blueprints=[
                FileBlueprint(
                    path="billing/calculator.py",
                    role="Billing calculation helpers",
                    required_imports=[],
                    symbols=[
                        SymbolSpec(
                            target_symbol="calc_discount",
                            signature="def calc_discount(amount: float) -> float:",
                            contract="Returns round(amount * 0.05, 2)",
                        )
                    ],
                )
            ],
            acceptance_criteria=["calc_discount(100.0) == 5.0"],
        ),
        stages=[
            StageGoal(
                stage_id=1,
                title="Billing Extension",
                objective="1. `billing/calculator.py`:\n   Add calc_discount(amount).",
                target_files=["billing/calculator.py"],
            )
        ],
        workspace_origin="external_repo",
        snapshot_count=4,
    )
    print(preview_text)
    assert "CLI SPEC & ARCHITECTURAL BLUEPRINT PREVIEW" in preview_text
    assert "Workspace Mode: external_repo (4 pre-existing file(s) snapshotted)" in preview_text
    assert "calc_discount" in preview_text
    print("=== PART 1 PASSED ===")


def test_part2_situation1_orchestrator_native_incremental() -> None:
    """
    Part 2 (Situation 1 — `orchestrator_native`):
    Seeds an orchestrator-created repository (with `.checkpoint/state.json` and `tests/test_stage_1.py`
    .. `tests/test_stage_3.py` from the Helios-One microgrid build) and runs an incremental feature
    addition WITHOUT `--clean`:
    - Adds method `effective_kw(self, loss_factor: float) -> float` to existing `InverterReading` in `domain/models.py`
    - Adds function `summarize_viable_count(readings: list[InverterReading], min_kw: float) -> int` to existing `services/analytics.py`
    - Updates `main.py` to print `Helios-One Net Dispatch: 190.0 kW (Viable: 2)`
    Verifies that all pre-existing methods/functions and all prior stage test suites remain intact!
    """
    print("\n================================================================")
    print("=== PART 2: SITUATION 1 (ORCHESTRATOR-NATIVE INCREMENTAL) ===")
    print("================================================================")
    clean_workspace()

    for sub in ("config", "data", "domain", "services", "tests"):
        (WORKER_WORKSPACE / sub).mkdir(parents=True, exist_ok=True)
        if sub in ("domain", "services", "tests"):
            (WORKER_WORKSPACE / sub / "__init__.py").write_text("", encoding="utf-8")

    (WORKER_WORKSPACE / "config" / "grid_profile.json").write_text(
        json.dumps({"grid_name": "Helios-One", "loss_factor": 0.05, "min_kw": 50.0}) + "\n",
        encoding="utf-8",
    )
    (WORKER_WORKSPACE / "data" / "inverters.csv").write_text(
        "inv_id,kw_output,temp_c\nINV-01,120.0,45.0\nINV-02,80.0,62.0\nINV-03,40.0,78.0\n",
        encoding="utf-8",
    )
    (WORKER_WORKSPACE / "domain" / "models.py").write_text(
        "class InverterReading:\n"
        "    def __init__(self, inv_id: str, kw_output: float, temp_c: float):\n"
        "        self.inv_id = str(inv_id)\n"
        "        self.kw_output = float(kw_output)\n"
        "        self.temp_c = float(temp_c)\n\n"
        "    def is_viable(self, min_kw: float) -> bool:\n"
        "        return self.kw_output >= min_kw and self.temp_c < 75.0\n",
        encoding="utf-8",
    )
    (WORKER_WORKSPACE / "services" / "ingestion.py").write_text(
        "import csv\nimport json\nfrom domain.models import InverterReading\n\n\n"
        "def load_grid_profile(path: str = 'config/grid_profile.json') -> dict:\n"
        "    with open(path, encoding='utf-8') as f:\n"
        "        return json.load(f)\n\n\n"
        "def load_readings(path: str = 'data/inverters.csv') -> list[InverterReading]:\n"
        "    with open(path, newline='', encoding='utf-8') as f:\n"
        "        return [InverterReading(r['inv_id'], float(r['kw_output']), float(r['temp_c'])) for r in csv.DictReader(f)]\n",
        encoding="utf-8",
    )
    (WORKER_WORKSPACE / "services" / "analytics.py").write_text(
        "from domain.models import InverterReading\n\n\n"
        "def compute_net_dispatch(readings: list[InverterReading], min_kw: float, loss_factor: float) -> float:\n"
        "    total_kw = sum(r.kw_output for r in readings if r.is_viable(min_kw))\n"
        "    return round(total_kw * (1.0 - loss_factor), 2)\n",
        encoding="utf-8",
    )
    (WORKER_WORKSPACE / "main.py").write_text(
        "from services.ingestion import load_grid_profile, load_readings\n"
        "from services.analytics import compute_net_dispatch\n\n"
        "profile = load_grid_profile('config/grid_profile.json')\n"
        "readings = load_readings('data/inverters.csv')\n"
        "net_kw = compute_net_dispatch(readings, profile['min_kw'], profile['loss_factor'])\n"
        "assert net_kw == 190.0\n"
        "print(f'Helios-One Net Dispatch: {net_kw} kW')\n",
        encoding="utf-8",
    )
    # Seed stage 1 & stage 2 pytest regression suites (orchestrator_native markers)
    (WORKER_WORKSPACE / "tests" / "test_stage_1.py").write_text(
        "from domain.models import InverterReading\n\n"
        "def test_stage1_model() -> None:\n"
        "    r = InverterReading('INV-01', 120.0, 45.0)\n"
        "    assert r.is_viable(50.0) is True\n",
        encoding="utf-8",
    )
    (WORKER_WORKSPACE / "tests" / "test_stage_2.py").write_text(
        "from services.ingestion import load_grid_profile, load_readings\n"
        "from services.analytics import compute_net_dispatch\n\n"
        "def test_stage2_services() -> None:\n"
        "    prof = load_grid_profile('config/grid_profile.json')\n"
        "    rds = load_readings('data/inverters.csv')\n"
        "    assert compute_net_dispatch(rds, prof['min_kw'], prof['loss_factor']) == 190.0\n",
        encoding="utf-8",
    )

    origin, snaps, next_stg = inspect_workspace_baseline()
    print(f"Detected workspace origin: {origin}, snapshots: {sorted(snaps.keys())}, next_stage_id: {next_stg}")
    assert origin == "orchestrator_native"
    assert next_stg == 3

    incremental_goal = """
    Extend the existing Helios-One solar microgrid project without removing any existing classes, methods, or functions:
    1. `domain/models.py`:
       Add a new method `effective_kw(self, loss_factor: float) -> float` to `InverterReading` that returns `round(self.kw_output * (1.0 - loss_factor), 2)`. Keep `__init__` and `is_viable` intact.
    2. `services/analytics.py`:
       Add a new function `count_viable(readings: list[InverterReading], min_kw: float) -> int` that returns the count of readings in `readings` where `r.is_viable(min_kw)` is True. Keep `compute_net_dispatch` intact.
    3. `main.py`:
       Import `load_grid_profile` and `load_readings` from `services.ingestion` and `compute_net_dispatch`, `count_viable` from `services.analytics`. Load the profile and readings, compute `net_kw = compute_net_dispatch(readings, profile['min_kw'], profile['loss_factor'])` and `viable = count_viable(readings, profile['min_kw'])`, assert `net_kw == 190.0` and `viable == 2`, and print the exact line:
       Helios-One Net Dispatch: 190.0 kW (Viable: 2)
    """

    profile = WorkerProfile(
        model="ibm/granite4.1:3b",
        context_limit="limited",
        strengths=["localized single-file edits", "writing clean functions and methods"],
        avoid=["architecture redesign", "deleting existing functions"],
        execution_mode="one_session_per_task",
    )

    history = run_project(
        goal=incremental_goal,
        architecture="Incremental extension of existing orchestrator-native multi-package workspace.",
        worker_profile=profile,
        max_cycles=2,
        emit_tests=True,
    )

    assert history[-1].outcome == "overall_success", f"Expected overall_success, got {history[-1].outcome}"

    models_syms = extract_all_ast_symbol_names((WORKER_WORKSPACE / "domain" / "models.py").read_text(encoding="utf-8"))
    analytics_syms = extract_all_ast_symbol_names((WORKER_WORKSPACE / "services" / "analytics.py").read_text(encoding="utf-8"))
    print("Updated domain/models.py symbols:", sorted(models_syms))
    print("Updated services/analytics.py symbols:", sorted(analytics_syms))

    assert {"InverterReading", "InverterReading.__init__", "InverterReading.is_viable", "InverterReading.effective_kw"}.issubset(models_syms)
    assert {"compute_net_dispatch", "count_viable"}.issubset(analytics_syms)
    assert (WORKER_WORKSPACE / "tests" / "test_stage_3.py").is_file(), "Expected incremental run to emit tests/test_stage_3.py!"

    completed = subprocess.run(
        [sys.executable, str(WORKER_WORKSPACE / "main.py")],
        cwd=WORKER_WORKSPACE,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, f"main.py failed:\n{completed.stderr}"
    assert completed.stdout.strip() == "Helios-One Net Dispatch: 190.0 kW (Viable: 2)"
    print("=== PART 2 (SITUATION 1: ORCHESTRATOR-NATIVE INCREMENTAL) PASSED ===")


def test_part3_situation2_external_foreign_repo_bugfix_and_extension() -> None:
    """
    Part 3 (Situation 2 — `external_repo`):
    Seeds a foreign repository NOT created by our system (no `.checkpoint/`, no `.staging/`,
    no `test_stage_*.py`) containing a bug in an existing function `compute_invoice_total`
    inside `billing/processor.py`, a sibling function `format_currency` that must NOT be
    broken, an unrelated module `audit/logger.py`, and an existing legacy test file.
    Verifies surgical AST extraction, bug repair, feature addition, and non-regression!
    """
    print("\n================================================================")
    print("=== PART 3: SITUATION 2 (EXTERNAL FOREIGN REPO MODIFICATION) ===")
    print("================================================================")
    clean_workspace()

    for sub in ("billing", "audit", "tests"):
        (WORKER_WORKSPACE / sub).mkdir(parents=True, exist_ok=True)
        (WORKER_WORKSPACE / sub / "__init__.py").write_text("", encoding="utf-8")

    (WORKER_WORKSPACE / "billing" / "rates.json").write_text(
        json.dumps({"tax_rate": 0.10, "service_fee": 15.0}) + "\n",
        encoding="utf-8",
    )
    # Seed foreign billing/processor.py with:
    # - `format_currency(amount: float) -> str` (existing working function — MUST BE PRESERVED)
    # - `compute_invoice_total(subtotal: float, tax_rate: float, service_fee: float) -> float`
    #   (has a bug: subtracts service_fee instead of adding it!)
    (WORKER_WORKSPACE / "billing" / "processor.py").write_text(
        "def format_currency(amount: float) -> str:\n"
        "    return f'${amount:.2f}'\n\n\n"
        "def compute_invoice_total(subtotal: float, tax_rate: float, service_fee: float) -> float:\n"
        "    return round(subtotal * (1.0 + tax_rate) - service_fee, 2)\n",
        encoding="utf-8",
    )
    unrelated_code = (
        "def audit_banner(client: str) -> str:\n"
        "    return f'[AUDIT-OK] {client.strip().upper()}'\n"
    )
    (WORKER_WORKSPACE / "audit" / "logger.py").write_text(unrelated_code, encoding="utf-8")
    (WORKER_WORKSPACE / "main.py").write_text(
        "import json\nfrom billing.processor import compute_invoice_total, format_currency\n\n"
        "with open('billing/rates.json', encoding='utf-8') as f:\n"
        "    rates = json.load(f)\n"
        "total = compute_invoice_total(200.0, rates['tax_rate'], rates['service_fee'])\n"
        "print(f'Invoice Total: {format_currency(total)}')\n",
        encoding="utf-8",
    )
    (WORKER_WORKSPACE / "tests" / "test_legacy_currency.py").write_text(
        "from billing.processor import format_currency\n"
        "from audit.logger import audit_banner\n\n"
        "def test_legacy_helpers() -> None:\n"
        "    assert format_currency(12.5) == '$12.50'\n"
        "    assert audit_banner('acme') == '[AUDIT-OK] ACME'\n",
        encoding="utf-8",
    )

    origin, snaps, _ = inspect_workspace_baseline()
    print(f"Detected workspace origin: {origin}, snapshots: {sorted(snaps.keys())}")
    assert origin == "external_repo"

    foreign_goal = """
    Modify the existing billing repository without breaking `format_currency` or `audit/logger.py`:
    1. `billing/processor.py`:
       Fix `compute_invoice_total(subtotal: float, tax_rate: float, service_fee: float) -> float` so it adds `service_fee` instead of subtracting it, returning `round(subtotal * (1.0 + tax_rate) + service_fee, 2)`. Keep `format_currency` intact.
    2. `main.py`:
       Load `billing/rates.json`, call `compute_invoice_total(200.0, rates['tax_rate'], rates['service_fee'])`, assert `total == 235.0`, and print `Invoice Total: $235.00`.
    """

    profile = WorkerProfile(
        model="ibm/granite4.1:3b",
        context_limit="limited",
        strengths=["fixing functions in existing repositories", "localized single-file edits"],
        avoid=["deleting unrelated functions", "modifying untouched files"],
        execution_mode="one_session_per_task",
    )

    history = run_project(
        goal=foreign_goal,
        architecture="External Python repository in workspace (billing/, audit/, main.py).",
        worker_profile=profile,
        max_cycles=2,
        emit_tests=True,
    )

    assert history[-1].outcome == "overall_success", f"Expected overall_success, got {history[-1].outcome}"

    # Verify untouched foreign file was preserved byte-for-byte
    assert (WORKER_WORKSPACE / "audit" / "logger.py").read_text(encoding="utf-8") == unrelated_code

    # Verify both format_currency and compute_invoice_total exist in billing/processor.py
    proc_syms = extract_all_ast_symbol_names((WORKER_WORKSPACE / "billing" / "processor.py").read_text(encoding="utf-8"))
    assert {"format_currency", "compute_invoice_total"}.issubset(proc_syms)

    completed = subprocess.run(
        [sys.executable, str(WORKER_WORKSPACE / "main.py")],
        cwd=WORKER_WORKSPACE,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, f"main.py failed:\n{completed.stderr}"
    assert completed.stdout.strip() == "Invoice Total: $235.00"

    pytest_run = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "tests"],
        cwd=WORKER_WORKSPACE,
        capture_output=True,
        text=True,
        check=False,
    )
    print("External repo pytest output:\n" + pytest_run.stdout.strip())
    assert pytest_run.returncode == 0, f"Pytest failed:\n{pytest_run.stdout}\n{pytest_run.stderr}"
    print("\n=== ALPHA14 ALL TESTS PASSED ===")


def main() -> None:
    test_part1_ast_impact_slicer_non_regression_and_preview()
    test_part2_situation1_orchestrator_native_incremental()
    test_part3_situation2_external_foreign_repo_bugfix_and_extension()


if __name__ == "__main__":
    main()

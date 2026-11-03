from pathlib import Path
import shutil
import subprocess
import sys

# Bootstrap repository root into sys.path so 'src' imports resolve regardless of PYTHONPATH
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.orchestration.graph import WORKER_WORKSPACE, run_staged_project
from src.orchestration.schemas import WorkerProfile


def clean_sandbox() -> None:
    WORKER_WORKSPACE.mkdir(parents=True, exist_ok=True)
    for child in sorted(WORKER_WORKSPACE.iterdir()):
        if child.name == ".git":
            continue
        if child.is_dir():
            shutil.rmtree(child, ignore_errors=True)
        else:
            child.unlink(missing_ok=True)


def main() -> None:
    clean_sandbox()

    profile = WorkerProfile(
        model="ibm/granite4.1:3b",
        context_limit="compressed",
        strengths=["localized single-file edits"],
        avoid=["architecture redesign"],
        execution_mode="surgical_context_routing",
    )

    goal = """
    Build a multi-package hydrological control system in sandbox/:
    1. `config/dam_profile.json`: JSON object with exact keys `{"station": "Dam-Alpha", "max_capacity_cms": 500, "spillway_threshold": 350}`.
    2. `data/telemetry.csv`: CSV file with header `timestamp,inflow_cms` and 3 rows: `08:00,320`, `12:00,380`, `16:00,290`.
    3. `domain/models.py`: Define `class TelemetryRecord` with `__init__(self, timestamp: str, inflow: float)` storing `self.timestamp = timestamp` and `self.inflow = float(inflow)`, and method `is_flood(self, threshold: float) -> bool` returning `self.inflow > threshold`.
    4. `services/analyzer.py`: Import `TelemetryRecord` from `domain.models` and define `analyze_flow(records: list[TelemetryRecord], threshold: float) -> int` returning `sum(1 for r in records if r.is_flood(threshold))`.
    5. `main.py`: Load `config/dam_profile.json` and `data/telemetry.csv`, build `TelemetryRecord` instances, call `analyze_flow(records, profile["spillway_threshold"])`, assert the count is `1`, and print the exact line:
       System OK: 1 flood events detected.
    """

    history = run_staged_project(
        goal=goal,
        architecture="Multi-package application: config/dam_profile.json, data/telemetry.csv, domain/models.py, services/analyzer.py, and main.py.",
        worker_profile=profile,
        max_stages=3,
        max_cycles_per_stage=2,
    )

    final = history[-1]
    assert final.outcome == "overall_success", f"Expected overall_success, got {final.outcome}"

    main_py = WORKER_WORKSPACE / "main.py"
    completed = subprocess.run(
        [sys.executable, str(main_py)],
        cwd=WORKER_WORKSPACE,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, f"main.py failed:\n{completed.stderr}"
    assert "System OK: 1 flood events detected." in completed.stdout, (
        f"Unexpected stdout: {completed.stdout!r}"
    )
    print("\n=== ALPHA11 SURGICAL CONTEXT & TOKEN BUDGET TEST PASSED ===")
    print(f"Output: {completed.stdout.strip()}")


if __name__ == "__main__":
    main()

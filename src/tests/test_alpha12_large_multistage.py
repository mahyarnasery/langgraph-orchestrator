from pathlib import Path
import shutil
import subprocess
import sys

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.harnesses.aider import build_inline_dependency_context
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
        context_limit="compressed_pyi_stubs",
        strengths=["atomic functions", "single class methods", "config assets"],
        avoid=["monolithic multi-file edits"],
        execution_mode="hierarchical_staged_alpha12",
    )

    goal = """
    Build an 8-file multi-package hydrological spillway & telemetry platform in the workspace:

    1. `config/basin_thresholds.json`:
       JSON object with exact keys and values:
       `{"basin": "Zagros-Basin", "base_elevation_m": 1200.0, "critical_head_m": 1250.0, "weir_coeff": 2.0}`.

    2. `data/sensor_readings.csv`:
       CSV file with header `station_id,water_level_m,gate_width_m,status` and exactly 4 rows:
       `ST-101,1254.0,10.0,ACTIVE`
       `ST-102,1248.5,8.0,ACTIVE`
       `ST-103,1259.0,10.0,ACTIVE`
       `ST-104,1265.0,10.0,MAINTENANCE`

    3. `domain/reading.py`:
       Define `class SensorReading` with three methods:
       - `__init__(self, station_id: str, water_level_m: float, gate_width_m: float, status: str) -> None`:
         stores `self.station_id = station_id`, `self.water_level_m = float(water_level_m)`,
         `self.gate_width_m = float(gate_width_m)`, and `self.status = status`.
       - `is_active(self) -> bool`:
         returns `self.status == "ACTIVE"`.
       - `overtop_head(self, critical_head_m: float) -> float`:
         if `not self.is_active()`, returns `0.0`; otherwise returns `max(0.0, self.water_level_m - float(critical_head_m))`.

    4. `domain/weir.py`:
       Import `SensorReading` from `domain.reading`. Define `class WeirModel` with two methods:
       - `__init__(self, critical_head_m: float, weir_coeff: float) -> None`:
         stores `self.critical_head_m = float(critical_head_m)` and `self.weir_coeff = float(weir_coeff)`.
       - `compute_discharge(self, reading: SensorReading) -> float`:
         computes `h = reading.overtop_head(self.critical_head_m)` and returns `round(self.weir_coeff * reading.gate_width_m * (h ** 1.5), 2)`.

    5. `services/loader.py`:
       Import `json`, `csv`, and `SensorReading` from `domain.reading`. Implement two standalone functions:
       - `load_basin_config(path: str) -> dict`:
         reads and returns the parsed JSON dictionary from `path`.
       - `load_sensor_readings(path: str) -> list[SensorReading]`:
         reads CSV rows from `path` using `csv.DictReader` and returns a list of
         `SensorReading(row["station_id"], float(row["water_level_m"]), float(row["gate_width_m"]), row["status"])`.

    6. `services/hydraulics.py`:
       Import `SensorReading` from `domain.reading` and `WeirModel` from `domain.weir`. Implement two standalone functions:
       - `filter_overflow_stations(readings: list[SensorReading], critical_head_m: float) -> list[str]`:
         returns `[r.station_id for r in readings if r.overtop_head(critical_head_m) > 0.0]`.
       - `calculate_total_discharge(readings: list[SensorReading], weir: WeirModel) -> float`:
         returns `round(sum(weir.compute_discharge(r) for r in readings), 2)`.

    7. `services/reporter.py`:
       Import `SensorReading` from `domain.reading`, `WeirModel` from `domain.weir`, and
       `filter_overflow_stations`, `calculate_total_discharge` from `services.hydraulics`.
       Implement two standalone functions:
       - `format_station_list(station_ids: list[str]) -> str`:
         returns `",".join(station_ids) if station_ids else "NONE"`.
       - `build_basin_summary(basin_name: str, readings: list[SensorReading], weir: WeirModel) -> str`:
         computes `active_count = sum(1 for r in readings if r.is_active())`,
         `overflow_ids = filter_overflow_stations(readings, weir.critical_head_m)`,
         `stations_str = format_station_list(overflow_ids)`,
         `total_q = calculate_total_discharge(readings, weir)`,
         and returns `f"{basin_name} | Active={active_count}/{len(readings)} | Overflow=[{stations_str}] | TotalQ={total_q:.1f} cms"`.

    8. `main.py`:
       Import `load_basin_config` and `load_sensor_readings` from `services.loader`,
       `WeirModel` from `domain.weir`, and `build_basin_summary` from `services.reporter`.
       Load `cfg = load_basin_config("config/basin_thresholds.json")` and
       `readings = load_sensor_readings("data/sensor_readings.csv")`,
       instantiate `weir = WeirModel(cfg["critical_head_m"], cfg["weir_coeff"])`,
       compute `summary = build_basin_summary(cfg["basin"], readings, weir)`,
       assert `summary == "Zagros-Basin | Active=3/4 | Overflow=[ST-101,ST-103] | TotalQ=700.0 cms"`,
       and print `summary`.
    """

    architecture = (
        "8-file multi-package Python hydrological platform: "
        "config/basin_thresholds.json, data/sensor_readings.csv, "
        "domain/reading.py, domain/weir.py, services/loader.py, "
        "services/hydraulics.py, services/reporter.py, and main.py."
    )

    history = run_staged_project(
        goal=goal,
        architecture=architecture,
        worker_profile=profile,
        max_stages=3,
        max_cycles_per_stage=2,
        emit_tests=True,
    )

    final = history[-1]
    assert final.outcome == "overall_success", f"Expected overall_success, got {final.outcome}"

    expected_files = [
        "config/basin_thresholds.json",
        "data/sensor_readings.csv",
        "domain/reading.py",
        "domain/weir.py",
        "services/loader.py",
        "services/hydraulics.py",
        "services/reporter.py",
        "main.py",
    ]
    for rel_path in expected_files:
        assert (WORKER_WORKSPACE / rel_path).is_file(), f"Missing expected file: {rel_path}"

    # Verify .pyi inline context compression across all 7 dependency files
    full_raw_size = sum(
        len((WORKER_WORKSPACE / rel_p).read_text(encoding="utf-8"))
        for rel_p in expected_files[:-1]
    )
    compressed_inline = build_inline_dependency_context(expected_files[:-1], WORKER_WORKSPACE)
    print(f"\n[CONTEXT METRICS] Raw 7-file size: {full_raw_size} chars | Inline .pyi + asset context: {len(compressed_inline)} chars")

    # Execute main.py and verify exact output
    completed = subprocess.run(
        [sys.executable, str(WORKER_WORKSPACE / "main.py")],
        cwd=WORKER_WORKSPACE,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, f"main.py failed:\n{completed.stderr}"
    expected_out = "Zagros-Basin | Active=3/4 | Overflow=[ST-101,ST-103] | TotalQ=700.0 cms"
    assert completed.stdout.strip() == expected_out, (
        f"Unexpected output: {completed.stdout.strip()!r} (expected {expected_out!r})"
    )

    # Run pytest on all emitted stage test suites
    env = dict(subprocess.os.environ, PYTHONPATH=str(WORKER_WORKSPACE))
    pytest_run = subprocess.run(
        [sys.executable, "-m", "pytest", "-v", str(WORKER_WORKSPACE / "tests")],
        cwd=WORKER_WORKSPACE,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    print("\n=== EMITTED PYTEST SUITE OUTPUT ===")
    print(pytest_run.stdout)
    assert pytest_run.returncode == 0, f"Emitted pytest suites failed:\n{pytest_run.stdout}\n{pytest_run.stderr}"

    print("\n=== ALPHA12 8-FILE MULTI-STAGE STRESS TEST PASSED ===")
    print(f"Total cycles executed across 3 stages: {len(history)}")
    print(f"Final Output: {completed.stdout.strip()}")


if __name__ == "__main__":
    main()

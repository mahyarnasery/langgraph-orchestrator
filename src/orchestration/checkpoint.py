from __future__ import annotations

import ast
from pathlib import Path

from .ledger import (
    extract_python_signatures,
    merge_staged_snippets_ast,
    record_ledger_entry,
    upsert_staged_snippet,
)
from .schemas import CheckpointData, CycleResult, Task, TaskResult
from .state import WorkflowState


REPO_ROOT = Path("/mnt/ai/langgraph-orchestrator").resolve()
WORKSPACE_ROOT = (REPO_ROOT / "sandbox").resolve()
STAGING_DIR = (WORKSPACE_ROOT / ".staging").resolve()
CHECKPOINT_DIR = (WORKSPACE_ROOT / ".checkpoint").resolve()
CHECKPOINT_FILE = (CHECKPOINT_DIR / "state.json").resolve()


def _normalize_rel_path(path: str) -> str:
    cleaned = path.strip().replace("\\", "/")
    while cleaned.startswith("./"):
        cleaned = cleaned[2:]
    cleaned = cleaned.removeprefix("sandbox/").strip("/")
    return cleaned


def _is_valid_python_source(source: str, allow_empty_init: bool = False) -> bool:
    if not source.strip():
        return allow_empty_init
    try:
        ast.parse(source)
        return True
    except SyntaxError:
        return False


def save_checkpoint(
    state: WorkflowState,
    completed_cycles: list[CycleResult] | None = None,
) -> Path:
    """Persist the current WorkflowState atomically to sandbox/.checkpoint/state.json."""
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    existing_cycles: list[CycleResult] = []
    if completed_cycles is not None:
        existing_cycles = list(completed_cycles)
    elif CHECKPOINT_FILE.is_file():
        try:
            prev = CheckpointData.model_validate_json(
                CHECKPOINT_FILE.read_text(encoding="utf-8")
            )
            existing_cycles = prev.completed_cycles
        except Exception:
            existing_cycles = []

    data = CheckpointData(
        cycle_id=state["cycle_id"],
        project_goal=state["project_goal"],
        architecture_context=state["architecture_context"],
        worker_profile=state["worker_profile"],
        previous_cycle=state.get("previous_cycle"),
        plan=state.get("plan"),
        task_batches=list(state.get("task_batches", [])),
        task_results=list(state.get("task_results", [])),
        ledger=list(state.get("ledger", [])),
        staged_files={
            _normalize_rel_path(k): dict(v)
            for k, v in state.get("staged_files", {}).items()
        },
        file_snapshots={
            _normalize_rel_path(k): v
            for k, v in state.get("file_snapshots", {}).items()
        },
        worker_attempts=dict(state.get("worker_attempts", {})),
        foreman_repair_attempts=state.get("foreman_repair_attempts", 0),
        rebuild_attempts=state.get("rebuild_attempts", 0),
        validation_result=state.get("validation_result"),
        foreman_review=state.get("foreman_review"),
        planner_review=state.get("planner_review"),
        rework_mode=state.get("rework_mode", "none"),
        phase=state.get("phase", "initial"),
        completed_cycles=existing_cycles,
    )

    tmp_file = CHECKPOINT_DIR / "state.json.tmp"
    tmp_file.write_text(data.model_dump_json(indent=2), encoding="utf-8")
    tmp_file.replace(CHECKPOINT_FILE)
    return CHECKPOINT_FILE


def load_checkpoint() -> CheckpointData | None:
    """Load the saved checkpoint from sandbox/.checkpoint/state.json if it exists."""
    if not CHECKPOINT_FILE.is_file():
        return None
    try:
        return CheckpointData.model_validate_json(
            CHECKPOINT_FILE.read_text(encoding="utf-8")
        )
    except Exception as exc:
        print(f"[WARN] Could not parse checkpoint file {CHECKPOINT_FILE}: {exc}")
        return None


def clear_checkpoint() -> None:
    """Remove any existing checkpoint file."""
    if CHECKPOINT_FILE.is_file():
        CHECKPOINT_FILE.unlink(missing_ok=True)


def reconcile_checkpoint_with_disk(checkpoint: CheckpointData) -> WorkflowState:
    """
    Compare a loaded CheckpointData against the physical files in `sandbox/` and
    `sandbox/.staging/`, recovering orphaned staged snippets and invalidating any
    task whose staged or target files were deleted, emptied, or corrupted on disk.
    """
    print("\n=== CHECKPOINT & DISK RECONCILIATION ===")
    ledger = list(checkpoint.ledger)
    cycle_id = checkpoint.cycle_id

    staged_files: dict[str, dict[str, str]] = {
        _normalize_rel_path(k): dict(v) for k, v in checkpoint.staged_files.items()
    }
    snapshots = {
        _normalize_rel_path(k): v for k, v in checkpoint.file_snapshots.items()
    }

    all_tasks_by_id: dict[str, Task] = {}
    file_task_counts: dict[str, int] = {}
    for batch in checkpoint.task_batches:
        for t in batch.tasks:
            all_tasks_by_id[t.task_id] = t
            for rf in t.relevant_files:
                norm_rf = _normalize_rel_path(rf)
                file_task_counts[norm_rf] = file_task_counts.get(norm_rf, 0) + 1

    verified_results: dict[str, TaskResult] = {}
    invalidated_tasks: list[str] = []
    recovered_tasks: list[str] = []

    # 1. Verify each recorded TaskResult against physical files on disk
    for res in checkpoint.task_results:
        task_obj = all_tasks_by_id.get(res.task_id)
        primary_file = (
            _normalize_rel_path(task_obj.relevant_files[0])
            if task_obj and task_obj.relevant_files
            else (
                _normalize_rel_path(res.files_changed[0])
                if res.files_changed
                else ""
            )
        )

        if res.status != "completed":
            invalidated_tasks.append(f"{res.task_id}(status={res.status})")
            if primary_file in staged_files:
                staged_files[primary_file].pop(res.task_id, None)
            continue

        if res.staged_file:
            staged_rel = _normalize_rel_path(res.staged_file)
            staged_abs = WORKSPACE_ROOT / staged_rel
            if staged_abs.is_file():
                snippet = staged_abs.read_text(encoding="utf-8").strip()
                if _is_valid_python_source(snippet):
                    if primary_file:
                        staged_files[primary_file] = upsert_staged_snippet(
                            staged_files.get(primary_file, {}),
                            res.task_id,
                            snippet,
                        )
                    verified_results[res.task_id] = res
                else:
                    invalidated_tasks.append(f"{res.task_id}(corrupted staged file {staged_rel})")
                    if primary_file in staged_files:
                        staged_files[primary_file].pop(res.task_id, None)
            else:
                invalidated_tasks.append(f"{res.task_id}(missing staged file {staged_rel})")
                if primary_file in staged_files:
                    staged_files[primary_file].pop(res.task_id, None)
        else:
            changed_paths = [
                _normalize_rel_path(p)
                for p in (res.files_changed or ([primary_file] if primary_file else []))
                if _normalize_rel_path(p)
            ]
            all_ok = bool(changed_paths)
            for rel_p in changed_paths:
                abs_p = WORKSPACE_ROOT / rel_p
                if not abs_p.is_file():
                    all_ok = False
                    break
                content = abs_p.read_text(encoding="utf-8")
                if rel_p.endswith(".py"):
                    if not _is_valid_python_source(
                        content, allow_empty_init=(abs_p.name == "__init__.py")
                    ):
                        all_ok = False
                        break
                elif not content.strip():
                    all_ok = False
                    break

            if all_ok:
                verified_results[res.task_id] = res
            else:
                invalidated_tasks.append(
                    f"{res.task_id}(missing/corrupted target file {','.join(changed_paths)})"
                )

    # 2. Scan active batch for orphaned valid `.staging/` files written right before an interrupt
    active_batch = checkpoint.task_batches[-1] if checkpoint.task_batches else None
    if active_batch is not None:
        for task in active_batch.tasks:
            if task.task_id in verified_results:
                continue
            if not task.relevant_files:
                continue
            primary_file = _normalize_rel_path(task.relevant_files[0])
            use_staging = (
                file_task_counts.get(primary_file, 0) > 1
                or primary_file in staged_files
            )
            if use_staging:
                safe_name = primary_file.replace("/", "__")
                staged_rel = f".staging/{task.task_id}__{safe_name}"
                staged_abs = WORKSPACE_ROOT / staged_rel
                if staged_abs.is_file():
                    snippet = staged_abs.read_text(encoding="utf-8").strip()
                    if _is_valid_python_source(snippet):
                        sigs = " ".join(extract_python_signatures(snippet))
                        if not task.target_symbol or task.target_symbol in sigs:
                            staged_files[primary_file] = upsert_staged_snippet(
                                staged_files.get(primary_file, {}),
                                task.task_id,
                                snippet,
                            )
                            verified_results[task.task_id] = TaskResult(
                                task_id=task.task_id,
                                status="completed",
                                summary=f"Recovered staged task {task.task_id} for {primary_file} from disk.",
                                files_changed=[primary_file],
                                staged_file=staged_rel,
                                errors=[],
                            )
                            recovered_tasks.append(task.task_id)

    # 3. Re-synchronize live AST-merged files on disk for any staged modules
    for mod_file, snippets_map in sorted(staged_files.items()):
        if not snippets_map:
            continue
        mod_abs = WORKSPACE_ROOT / mod_file
        if not mod_abs.is_file() or not mod_abs.read_text(encoding="utf-8").strip():
            merged_live = merge_staged_snippets_ast(
                target_filename=mod_file,
                base_source=snapshots.get(mod_file, ""),
                ordered_snippets=list(snippets_map.items()),
            )
            if merged_live.strip():
                mod_abs.parent.mkdir(parents=True, exist_ok=True)
                mod_abs.write_text(merged_live, encoding="utf-8")
                print(f"[RECONCILE] Restored live AST merge for sandbox/{mod_file}")

    # 4. Determine reconciled workflow phase
    reconciled_phase = checkpoint.phase
    if checkpoint.plan is None:
        reconciled_phase = "initial"
    elif not checkpoint.task_batches:
        reconciled_phase = "planned"
    elif active_batch is not None:
        active_task_ids = [t.task_id for t in active_batch.tasks]
        pending_active = [
            tid for tid in active_task_ids if tid not in verified_results
        ]
        if pending_active:
            reconciled_phase = (
                "corrective_task_ready"
                if len(checkpoint.task_batches) > 1
                else "task_ready"
            )
            print(
                f"[RECONCILE] Active batch '{active_batch.batch_id}' has "
                f"{len( verified_results )}/{len(active_task_ids)} verified task(s); "
                f"pending tasks to run: {pending_active} -> routing to worker."
            )
        else:
            # All tasks in active batch are verified on disk. Check if any staged module
            # still needs final assembly or if we can keep the later checkpointed phase.
            if reconciled_phase in ("task_ready", "corrective_task_ready"):
                reconciled_phase = "worker_complete"
            print(
                f"[RECONCILE] All {len(active_task_ids)} task(s) in batch "
                f"'{active_batch.batch_id}' verified on disk -> resuming at phase '{reconciled_phase}'."
            )

    summary_parts = [
        f"phase={checkpoint.phase}->{reconciled_phase}",
        f"verified_tasks={ sorted(verified_results.keys()) }",
    ]
    if invalidated_tasks:
        summary_parts.append(f"invalidated={invalidated_tasks}")
    if recovered_tasks:
        summary_parts.append(f"recovered={recovered_tasks}")

    ledger = record_ledger_entry(
        ledger,
        cycle_id=cycle_id,
        layer="checkpoint",
        action="reconcile_disk",
        target=reconciled_phase,
        status="warning" if invalidated_tasks else "success",
        detail="; ".join(summary_parts),
    )

    return {
        "cycle_id": checkpoint.cycle_id,
        "project_goal": checkpoint.project_goal,
        "architecture_context": checkpoint.architecture_context,
        "worker_profile": checkpoint.worker_profile,
        "previous_cycle": checkpoint.previous_cycle,
        "plan": checkpoint.plan,
        "task_batches": list(checkpoint.task_batches),
        "task_results": list(verified_results.values()),
        "ledger": ledger,
        "staged_files": staged_files,
        "file_snapshots": snapshots,
        "worker_attempts": dict(checkpoint.worker_attempts),
        "foreman_repair_attempts": checkpoint.foreman_repair_attempts,
        "rebuild_attempts": checkpoint.rebuild_attempts,
        "validation_result": (
            checkpoint.validation_result
            if reconciled_phase not in ("task_ready", "corrective_task_ready", "worker_complete")
            else None
        ),
        "foreman_review": (
            checkpoint.foreman_review
            if reconciled_phase.startswith("foreman_review_")
            else None
        ),
        "planner_review": checkpoint.planner_review,
        "rework_mode": checkpoint.rework_mode,
        "phase": reconciled_phase,
    }

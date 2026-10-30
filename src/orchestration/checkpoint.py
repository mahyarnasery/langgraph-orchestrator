from __future__ import annotations

import ast
from pathlib import Path

from .ledger import (
    ensure_class_wrapper,
    extract_all_ast_symbol_names,
    merge_staged_snippets_ast,
    record_ledger_entry,
    upsert_staged_snippet,
)
from .schemas import CheckpointData, CycleResult, StageGoal, Task, TaskResult
from .state import WorkflowState
from .workspace import (
    get_checkpoint_dir,
    get_checkpoint_file,
    get_workspace_root,
    normalize_workspace_rel_path,
)


def _normalize_rel_path(path: str) -> str:
    return normalize_workspace_rel_path(path)


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
    *,
    completed_cycles: list[CycleResult] | None = None,
    stages: list[StageGoal] | None = None,
    current_stage_index: int | None = None,
    completed_stage_ids: list[int] | None = None,
) -> Path:
    """
    Persist the current WorkflowState atomically to `<workspace>/.checkpoint/state.json`,
    preserving multi-stage metadata (`stages`, `current_stage_index`, `completed_stage_ids`,
    `completed_cycles`) unless explicitly updated.
    """
    ckpt_dir = get_checkpoint_dir()
    ckpt_file = get_checkpoint_file()

    prev: CheckpointData | None = None
    if ckpt_file.is_file():
        try:
            prev = CheckpointData.model_validate_json(
                ckpt_file.read_text(encoding="utf-8")
            )
        except Exception:
            prev = None

    resolved_cycles = (
        list(completed_cycles)
        if completed_cycles is not None
        else (list(prev.completed_cycles) if prev else [])
    )
    resolved_stages = (
        list(stages)
        if stages is not None
        else (list(prev.stages) if prev else [])
    )
    resolved_stage_idx = (
        current_stage_index
        if current_stage_index is not None
        else (prev.current_stage_index if prev else 0)
    )
    resolved_completed_stage_ids = (
        list(completed_stage_ids)
        if completed_stage_ids is not None
        else (list(prev.completed_stage_ids) if prev else [])
    )

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
        completed_cycles=resolved_cycles,
        stages=resolved_stages,
        current_stage_index=resolved_stage_idx,
        completed_stage_ids=resolved_completed_stage_ids,
    )

    tmp_file = ckpt_dir / "state.json.tmp"
    tmp_file.write_text(data.model_dump_json(indent=2), encoding="utf-8")
    tmp_file.replace(ckpt_file)
    return ckpt_file




def update_checkpoint_stage_metadata(
    *,
    completed_cycles: list[CycleResult],
    stages: list[StageGoal],
    current_stage_index: int,
    completed_stage_ids: list[int],
) -> None:
    """Update multi-stage progress fields on the existing checkpoint file without re-running disk reconciliation."""
    ckpt_dir = get_checkpoint_dir()
    ckpt_file = get_checkpoint_file()
    if not ckpt_file.is_file():
        return
    try:
        data = CheckpointData.model_validate_json(ckpt_file.read_text(encoding="utf-8"))
        data.completed_cycles = list(completed_cycles)
        data.stages = list(stages)
        data.current_stage_index = current_stage_index
        data.completed_stage_ids = list(completed_stage_ids)
        tmp_file = ckpt_dir / "state.json.tmp"
        tmp_file.write_text(data.model_dump_json(indent=2), encoding="utf-8")
        tmp_file.replace(ckpt_file)
    except Exception as exc:
        print(f"[WARN] Could not update stage metadata in checkpoint: {exc}")


def load_checkpoint() -> CheckpointData | None:
    """Load the saved checkpoint from `<workspace>/.checkpoint/state.json` if it exists."""
    ckpt_file = get_checkpoint_file()
    if not ckpt_file.is_file():
        return None
    try:
        return CheckpointData.model_validate_json(
            ckpt_file.read_text(encoding="utf-8")
        )
    except Exception as exc:
        print(f"[WARN] Could not parse checkpoint file {ckpt_file}: {exc}")
        return None


def clear_checkpoint() -> None:
    """Remove any existing checkpoint file in the active workspace."""
    ckpt_file = get_checkpoint_file()
    if ckpt_file.is_file():
        ckpt_file.unlink(missing_ok=True)


def reconcile_checkpoint_with_disk(checkpoint: CheckpointData) -> WorkflowState:
    """
    Compare a loaded CheckpointData against the physical files in the active
    workspace and its `.staging/` directory, recovering orphaned staged snippets
    and invalidating any task whose staged or target files were deleted, emptied,
    or corrupted on disk.
    """
    workspace_root = get_workspace_root()
    print(f"\n=== CHECKPOINT & DISK RECONCILIATION ({workspace_root}) ===")
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

    for res in checkpoint.task_results:
        task_obj = all_tasks_by_id.get(res.task_id)
        target_sym = task_obj.target_symbol if task_obj else None
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
            staged_abs = workspace_root / staged_rel
            if staged_abs.is_file():
                raw_snippet = staged_abs.read_text(encoding="utf-8").strip()
                snippet = ensure_class_wrapper(raw_snippet, target_sym).strip()
                if _is_valid_python_source(snippet):
                    if primary_file:
                        staged_files[primary_file] = upsert_staged_snippet(
                            staged_files.get(primary_file, {}),
                            res.task_id,
                            snippet,
                            target_symbol=target_sym,
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
                abs_p = workspace_root / rel_p
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
                staged_abs = workspace_root / staged_rel
                if staged_abs.is_file():
                    raw_snippet = staged_abs.read_text(encoding="utf-8").strip()
                    snippet = ensure_class_wrapper(raw_snippet, task.target_symbol).strip()
                    if _is_valid_python_source(snippet):
                        defined_syms = extract_all_ast_symbol_names(snippet)
                        if not task.target_symbol or task.target_symbol in defined_syms:
                            staged_files[primary_file] = upsert_staged_snippet(
                                staged_files.get(primary_file, {}),
                                task.task_id,
                                snippet,
                                target_symbol=task.target_symbol,
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

    for mod_file, snippets_map in sorted(staged_files.items()):
        if not snippets_map:
            continue
        mod_abs = workspace_root / mod_file
        if not mod_abs.is_file() or not mod_abs.read_text(encoding="utf-8").strip():
            merged_live = merge_staged_snippets_ast(
                target_filename=mod_file,
                base_source=snapshots.get(mod_file, ""),
                ordered_snippets=list(snippets_map.items()),
            )
            if merged_live.strip():
                mod_abs.parent.mkdir(parents=True, exist_ok=True)
                mod_abs.write_text(merged_live, encoding="utf-8")
                print(f"[RECONCILE] Restored live AST merge for {mod_file}")

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
                f"{len(verified_results)}/{len(active_task_ids)} verified task(s); "
                f"pending tasks to run: {pending_active} -> routing to worker."
            )
        else:
            if reconciled_phase in ("task_ready", "corrective_task_ready"):
                reconciled_phase = "worker_complete"
            print(
                f"[RECONCILE] All {len(active_task_ids)} task(s) in batch "
                f"'{active_batch.batch_id}' verified on disk -> resuming at phase '{reconciled_phase}'."
            )

    summary_parts = [
        f"phase={checkpoint.phase}->{reconciled_phase}",
        f"verified_tasks={sorted(verified_results.keys())}",
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

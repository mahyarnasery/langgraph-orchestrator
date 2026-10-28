from __future__ import annotations

import ast
from pathlib import Path

from langgraph.graph import END, StateGraph

from src.agents.foreman import ForemanAgent
from src.agents.planner import PlannerAgent
from src.harnesses.aider import AiderHarness, STAGING_DIR, list_workspace_py_files

from .ledger import (
    extract_python_signatures,
    format_ledger_for_prompt,
    merge_staged_snippets_ast,
    record_ledger_entry,
)
from .schemas import (
    CycleResult,
    ForemanReview,
    Plan,
    PlannerReview,
    TaskBatch,
    TaskResult,
    ValidationResult,
    WorkerProfile,
)
from .state import WorkflowState
from .validator import validate_batch

REPO_ROOT = Path("/mnt/ai/langgraph-orchestrator").resolve()
WORKER_WORKSPACE = (REPO_ROOT / "sandbox").resolve()
MAX_FOREMAN_REPAIRS = 2
MAX_WORKER_REBUILDS = 3


def _sandbox_repo_path(path: str) -> str:
    """Convert a logical TaskBatch path into its physical repository path inside sandbox."""
    normalized = path.removeprefix("sandbox/")
    if normalized in ("", ".", ".."):
        raise ValueError(f"Invalid Worker path: {path}")
    return f"sandbox/{normalized}"


def _build_repository_map() -> str:
    """Build a compact AST-based map of existing files, classes, and functions in sandbox."""
    py_files = list_workspace_py_files()
    if not py_files:
        return "[No Python files currently in sandbox workspace]"

    entries: list[str] = []
    for py_file in py_files:
        rel_path = str(py_file.relative_to(WORKER_WORKSPACE))
        try:
            source = py_file.read_text(encoding="utf-8")
            if not source.strip():
                entries.append(f"- {rel_path} (empty file)")
                continue

            sigs = extract_python_signatures(source)
            summary = "; ".join(sigs) if sigs else f"{len(source.splitlines())} lines"
            entries.append(f"- {rel_path}: {summary}")
        except Exception:
            entries.append(f"- {rel_path} (unparsed)")

    return "Existing Workspace Repository Map:\n" + "\n".join(entries)


def _enrich_architecture_context(architecture_context: str) -> str:
    repo_map = _build_repository_map()
    return f"{architecture_context}\n\n{repo_map}"


def _prepare_batch_and_workspace(
    batch: TaskBatch,
    existing_snapshots: dict[str, str],
) -> tuple[TaskBatch, dict[str, str]]:
    """
    Normalize batch relevant_files and execution_order, pre-create missing
    files in sandbox, and snapshot pre-batch file contents for rollback support.
    """
    from .tools import create_file, read_file

    union_files = sorted(
        {
            path.removeprefix("sandbox/")
            for task in batch.tasks
            for path in task.relevant_files
            if path not in ("", ".", "..")
        }
        | {
            path.removeprefix("sandbox/")
            for path in batch.relevant_files
            if path not in ("", ".", "..")
        }
    )
    batch.relevant_files = union_files

    for task in batch.tasks:
        task.relevant_files = [
            p.removeprefix("sandbox/")
            for p in (task.relevant_files or union_files)
            if p not in ("", ".", "..")
        ]

    if len(batch.tasks) > 1:
        filtered_tasks = [
            t
            for t in batch.tasks
            if not (
                t.scope.strip().lower()
                in ("file creation", "create file", "empty file creation")
                and len(t.acceptance_criteria) == 1
                and "exists" in t.acceptance_criteria[0].lower()
            )
        ]
        if filtered_tasks:
            batch.tasks = filtered_tasks

    valid_ids = {t.task_id for t in batch.tasks}
    for task in batch.tasks:
        task.dependencies = [
            dep for dep in task.dependencies if dep in valid_ids and dep != task.task_id
        ]
    ordered_ids = [tid for tid in batch.execution_order if tid in valid_ids]
    for task in batch.tasks:
        if task.task_id not in ordered_ids:
            ordered_ids.append(task.task_id)
    batch.execution_order = ordered_ids

    snapshots = dict(existing_snapshots)

    print("\n=== FOREMAN REPOSITORY PREPARATION ===")
    for path in batch.relevant_files:
        physical_path = _sandbox_repo_path(path)
        try:
            res = read_file.func(path=physical_path)
            snapshots.setdefault(path, res.get("content", ""))
            print(f"{physical_path} already exists")
        except FileNotFoundError:
            create_file.func(path=physical_path, content="")
            snapshots.setdefault(path, "")
            print(f"Created {physical_path}")
        except Exception as exc:
            print(f"Warning: could not read {physical_path}: {exc}")
            try:
                create_file.func(path=physical_path, content="")
                snapshots.setdefault(path, "")
                print(f"Created {physical_path} (after read error)")
            except Exception as create_exc:
                print(f"Error creating {physical_path}: {create_exc}")

    return batch, snapshots


def _gather_repository_context(batch: TaskBatch, task_results: list[TaskResult]) -> str:
    """Inspect the sandbox repository files and construct textual evidence."""
    from .tools import read_file

    relevant_files = sorted(
        set(batch.relevant_files)
        | {
            file_path.removeprefix("sandbox/")
            for result in task_results
            for file_path in result.files_changed
        }
    )

    repository_evidence: list[str] = []
    for path in relevant_files:
        physical_path = _sandbox_repo_path(path)
        try:
            result = read_file.func(path=physical_path)
            content = result.get("content", "")
            content_display = content if content.strip() else "[EMPTY FILE - 0 bytes]"
            repository_evidence.append(f"FILE: {path}\n~~~python\n{content_display}\n~~~")
        except Exception as exc:
            repository_evidence.append(
                f"FILE: {path}\nREAD_ERROR: {type(exc).__name__}: {exc}"
            )

    return "\n\n".join(repository_evidence)


def _gather_full_repository_context() -> str:
    """Gather context from all user Python files in the sandbox for Planner review."""
    from .tools import read_file

    repository_evidence: list[str] = []
    for py_file in list_workspace_py_files():
        try:
            relative_path = py_file.relative_to(WORKER_WORKSPACE)
            logical_path = str(relative_path)
            result = read_file.func(path=f"sandbox/{logical_path}")
            content = result.get("content", "")
            content_display = content if content.strip() else "[EMPTY FILE - 0 bytes]"
            repository_evidence.append(
                f"FILE: {logical_path}\n~~~python\n{content_display}\n~~~"
            )
        except Exception as exc:
            repository_evidence.append(
                f"FILE: {logical_path}\nREAD_ERROR: {type(exc).__name__}: {exc}"
            )

    return (
        "\n\n".join(repository_evidence)
        if repository_evidence
        else "[No Python files in sandbox]"
    )


def build_planner() -> PlannerAgent:
    return PlannerAgent()


def build_foreman() -> ForemanAgent:
    return ForemanAgent()


def build_worker() -> AiderHarness:
    return AiderHarness()


# ============================================================================
# 1. Planner — Blueprinting
# ============================================================================


def planner_node(state: WorkflowState) -> dict:
    planner = build_planner()
    enriched_arch = _enrich_architecture_context(state["architecture_context"])
    ledger_ctx = format_ledger_for_prompt(state["ledger"], state["previous_cycle"])

    plan = planner.run(
        project_goal=state["project_goal"],
        architecture_context=enriched_arch,
        worker_profile=state["worker_profile"],
        previous_cycle=state["previous_cycle"],
        ledger_context=ledger_ctx,
    )

    if not isinstance(plan, Plan):
        raise TypeError("PlannerAgent.run() must return Plan.")

    print("\n=== PLANNER RESULT ===")
    print(plan.model_dump_json(indent=2))

    updated_ledger = record_ledger_entry(
        state["ledger"],
        cycle_id=state["cycle_id"],
        layer="planner",
        action="create_plan",
        target=", ".join(plan.affected_areas) or "workspace",
        status="success",
        detail=f"Planned {len(plan.acceptance_criteria)} acceptance criteria: {plan.objective}",
    )

    return {
        "plan": plan,
        "ledger": updated_ledger,
        "phase": "planned",
    }


# ============================================================================
# 2. Foreman — Task Atomization & Preparation
# ============================================================================


def foreman_node(state: WorkflowState) -> dict:
    plan = state["plan"]
    if plan is None:
        raise RuntimeError("Planner output missing.")

    foreman = build_foreman()
    enriched_arch = _enrich_architecture_context(state["architecture_context"])
    ledger_ctx = format_ledger_for_prompt(state["ledger"], state["previous_cycle"])

    batch = foreman.create_batch(
        project_goal=state["project_goal"],
        architecture_context=enriched_arch,
        worker_profile=state["worker_profile"],
        previous_cycle=state["previous_cycle"],
        plan=plan,
        ledger_context=ledger_ctx,
    )

    if not isinstance(batch, TaskBatch):
        raise TypeError("ForemanAgent.create_batch() must return TaskBatch.")

    batch, snapshots = _prepare_batch_and_workspace(
        batch, state.get("file_snapshots", {})
    )

    print("\n=== FOREMAN RESULT ===")
    print(batch.model_dump_json(indent=2))

    task_desc = ", ".join(
        f"{t.task_id}({t.target_symbol or ','.join(t.relevant_files)})"
        for t in batch.tasks
    )
    updated_ledger = record_ledger_entry(
        state["ledger"],
        cycle_id=state["cycle_id"],
        layer="foreman",
        action="atomize_tasks",
        target=batch.batch_id,
        status="success",
        detail=f"Atomized plan into {len(batch.tasks)} task(s): {task_desc}",
    )

    return {
        "task_batches": [batch],
        "file_snapshots": snapshots,
        "ledger": updated_ledger,
        "phase": "task_ready",
    }


# ============================================================================
# 3. Worker — Atomic Execution (Direct or Staged)
# ============================================================================


def worker_node(state: WorkflowState) -> dict:
    if not state["task_batches"]:
        raise RuntimeError("No TaskBatch received.")

    batch = state["task_batches"][-1]
    worker = build_worker()

    print("\n=== WORKER SESSION START ===")
    print(f"Batch: {batch.batch_id}")

    results = worker.execute(
        batch,
        existing_staged=state.get("staged_files", {}),
        file_snapshots=state.get("file_snapshots", {}),
        project_goal=state["project_goal"],
    )

    print("\n=== WORKER SESSION COMPLETE ===")

    attempts = dict(state["worker_attempts"])
    for result in results:
        attempts[result.task_id] = attempts.get(result.task_id, 0) + 1

    merged_by_id = {r.task_id: r for r in state["task_results"]}
    for result in results:
        merged_by_id[result.task_id] = result

    staged_files: dict[str, dict[str, str]] = {
        k: dict(v) for k, v in state.get("staged_files", {}).items()
    }
    tasks_by_id = {t.task_id: t for t in batch.tasks}

    updated_ledger = list(state["ledger"])
    for result in results:
        task_obj = tasks_by_id.get(result.task_id)
        target_file = (
            task_obj.relevant_files[0].removeprefix("sandbox/")
            if task_obj and task_obj.relevant_files
            else "unknown"
        )
        if result.staged_file:
            staged_path = WORKER_WORKSPACE / result.staged_file
            if staged_path.is_file():
                snippet = staged_path.read_text(encoding="utf-8").strip()
                if snippet:
                    staged_files.setdefault(target_file, {})[result.task_id] = snippet

        symbol_label = (
            f"{target_file}::{task_obj.target_symbol}"
            if task_obj and task_obj.target_symbol
            else target_file
        )
        updated_ledger = record_ledger_entry(
            updated_ledger,
            cycle_id=state["cycle_id"],
            layer="worker",
            action="execute_task",
            target=f"{result.task_id} ({symbol_label})",
            status="success" if result.status == "completed" else "warning",
            detail=result.summary,
        )

    return {
        "task_results": list(merged_by_id.values()),
        "worker_attempts": attempts,
        "staged_files": staged_files,
        "ledger": updated_ledger,
        "phase": "worker_complete",
    }


# ============================================================================
# 4. Foreman Assembler — Stitch Staged Atomic Functions into Coherent Modules
# ============================================================================


def foreman_assemble_node(state: WorkflowState) -> dict:
    """
    When a file was split into multiple function-level tasks in `.staging/`,
    Foreman assembles the individual snippets into the final target module.
    """
    from .tools import write_file

    staged_files = state.get("staged_files", {})
    if not staged_files:
        return {"phase": "assembled"}

    foreman = build_foreman()
    plan = state["plan"]
    snapshots = state.get("file_snapshots", {})
    updated_ledger = list(state["ledger"])
    ledger_ctx = format_ledger_for_prompt(updated_ledger, state["previous_cycle"])

    print("\n=== FOREMAN MODULE ASSEMBLY ===")

    for target_file, task_snippets in sorted(staged_files.items()):
        if not task_snippets:
            continue

        ast_merged = merge_staged_snippets_ast(
            target_filename=target_file,
            base_source=snapshots.get(target_file, ""),
            ordered_snippets=list(task_snippets.items()),
        )
        expected_sigs = extract_python_signatures(ast_merged)
        expected_fn_names = [
            s.split("(")[0].replace("def ", "").replace("class ", "").split("[")[0].strip()
            for s in expected_sigs
        ]

        final_code = ast_merged
        assembly_mode = "AST-deterministic"

        if len(task_snippets) > 1 and plan is not None:
            try:
                assembled = foreman.assemble_module(
                    target_file=target_file,
                    project_goal=state["project_goal"],
                    plan=plan,
                    staged_snippets=task_snippets,
                    ast_merged_draft=ast_merged,
                    ledger_context=ledger_ctx,
                )
                candidate = assembled.assembled_code.strip()
                if candidate:
                    ast.parse(candidate)
                    missing_symbols = [
                        name for name in expected_fn_names if name not in candidate
                    ]
                    if not missing_symbols:
                        final_code = candidate
                        assembly_mode = f"Foreman-LLM + AST verified ({assembled.notes or 'clean'})"
                    else:
                        print(
                            f"Foreman assembly omitted {missing_symbols}; "
                            f"retaining complete AST merge for {target_file}."
                        )
            except Exception as exc:
                print(
                    f"Foreman LLM assembly fallback to AST merge for {target_file}: {exc}"
                )

        physical_path = _sandbox_repo_path(target_file)
        write_file.func(path=physical_path, content=final_code)
        print(
            f"Assembled {physical_path} from {len(task_snippets)} staged snippet(s) "
            f"[{assembly_mode}]"
        )

        updated_ledger = record_ledger_entry(
            updated_ledger,
            cycle_id=state["cycle_id"],
            layer="assembler",
            action="assemble_module",
            target=target_file,
            status="success",
            detail=(
                f"Assembled {len(task_snippets)} snippet(s) "
                f"({', '.join(task_snippets.keys())}) -> { '; '.join(expected_sigs)}"
            ),
        )

    return {
        "ledger": updated_ledger,
        "phase": "assembled",
    }


# ============================================================================
# 5. Validator
# ============================================================================


def validator_node(state: WorkflowState) -> dict:
    if not state["task_batches"]:
        raise RuntimeError("No TaskBatch available for validation.")

    primary_batch = state["task_batches"][0]
    validation = validate_batch(
        batch=primary_batch,
        task_results=state["task_results"],
    )

    if not isinstance(validation, ValidationResult):
        raise TypeError("validate_batch() must return ValidationResult.")

    print("\n=== VALIDATOR RESULT ===")
    print(validation.model_dump_json(indent=2))

    detail = (
        f"All {len(validation.checks)} checks passed."
        if validation.passed
        else f"Failed with {len(validation.errors)} error(s): {'; '.join(validation.errors[:2])}"
    )
    updated_ledger = record_ledger_entry(
        state["ledger"],
        cycle_id=state["cycle_id"],
        layer="validator",
        action="validate_batch",
        target=", ".join(validation.files_checked),
        status="success" if validation.passed else "failure",
        detail=detail,
    )

    return {
        "validation_result": validation,
        "ledger": updated_ledger,
        "phase": "validated",
    }


# ============================================================================
# 6. Foreman — Review, Self-Repair & Corrective Rebuild
# ============================================================================


def foreman_review_node(state: WorkflowState) -> dict:
    if not state["task_batches"]:
        raise RuntimeError("No TaskBatch available for Foreman review.")
    if not state["task_results"]:
        raise RuntimeError("No TaskResults available for Foreman review.")

    validation = state["validation_result"]
    if validation is None:
        raise RuntimeError("ValidationResult missing before Foreman review.")

    plan = state["plan"]
    if plan is None:
        raise RuntimeError("Planner output missing before Foreman review.")

    batch = state["task_batches"][-1]
    repository_context = _gather_repository_context(
        state["task_batches"][0], state["task_results"]
    )
    enriched_arch = _enrich_architecture_context(state["architecture_context"])
    ledger_ctx = format_ledger_for_prompt(state["ledger"], state["previous_cycle"])

    foreman = build_foreman()
    review = foreman.review(
        project_goal=state["project_goal"],
        architecture_context=enriched_arch,
        plan=plan,
        batch=batch,
        task_results=state["task_results"],
        validation_result=validation,
        repository_context=repository_context,
        worker_attempts=state["worker_attempts"],
        foreman_repair_attempts=state["foreman_repair_attempts"],
        max_foreman_repairs=MAX_FOREMAN_REPAIRS,
        ledger_context=ledger_ctx,
    )

    if not isinstance(review, ForemanReview):
        raise TypeError("ForemanAgent.review() must return ForemanReview.")

    print("\n=== FOREMAN REVIEW ===")
    print(review.model_dump_json(indent=2))

    updated_ledger = record_ledger_entry(
        state["ledger"],
        cycle_id=state["cycle_id"],
        layer="foreman",
        action=f"review_{review.decision}",
        target=batch.batch_id,
        status="success" if review.decision == "accept" else "warning",
        detail=review.summary,
    )

    return {
        "foreman_review": review,
        "ledger": updated_ledger,
        "phase": f"foreman_review_{review.decision}",
    }


def foreman_self_repair_node(state: WorkflowState) -> dict:
    """Foreman executes direct localized repairs with exception safety."""
    from .tools import edit_file, write_file

    review = state["foreman_review"]
    print("\n=== FOREMAN DIRECT SELF-REPAIR ===")

    repaired_files: list[str] = []
    repair_errors: list[str] = []

    for repair in review.direct_repairs:
        physical_path = _sandbox_repo_path(repair.path)
        print(f"Applying Foreman self-repair to {physical_path} ({repair.action})")
        try:
            if repair.action == "edit":
                if not repair.target_text:
                    raise ValueError(
                        "Foreman 'edit' repair requires non-empty target_text."
                    )
                edit_file.func(
                    path=physical_path,
                    old_text=repair.target_text,
                    new_text=repair.replacement_text,
                )
            else:
                if not repair.content.strip():
                    raise ValueError(
                        "Foreman 'overwrite' repair requires non-empty content."
                    )
                write_file.func(
                    path=physical_path,
                    content=repair.content,
                )
            repaired_files.append(repair.path.removeprefix("sandbox/"))
        except Exception as exc:
            err_msg = f"Foreman self-repair failed on {repair.path}: {type(exc).__name__}: {exc}"
            print(f"WARNING: {err_msg}")
            repair_errors.append(err_msg)

    status = "completed" if repaired_files and not repair_errors else "failure"
    repair_id = f"foreman_self_repair_{state['foreman_repair_attempts'] + 1}"
    repair_task_result = TaskResult(
        task_id=repair_id,
        status=status,
        summary=f"Foreman direct repair: {', '.join(review.repair_instructions or ['self-repaired'])}",
        files_changed=repaired_files,
        errors=repair_errors,
    )

    updated_ledger = record_ledger_entry(
        state["ledger"],
        cycle_id=state["cycle_id"],
        layer="foreman",
        action="self_repair",
        target=", ".join(repaired_files) or "none",
        status="success" if status == "completed" else "failure",
        detail=repair_task_result.summary,
    )

    return {
        "task_results": state["task_results"] + [repair_task_result],
        "foreman_repair_attempts": state["foreman_repair_attempts"] + 1,
        "ledger": updated_ledger,
        "phase": "foreman_repaired",
    }


def foreman_corrective_node(state: WorkflowState) -> dict:
    """Foreman rolls back corrupted files (if requested) and creates a corrective TaskBatch."""
    from .tools import write_file

    foreman = build_foreman()
    batch = state["task_batches"][-1]
    review = state["foreman_review"]
    snapshots = state.get("file_snapshots", {})
    ledger_ctx = format_ledger_for_prompt(state["ledger"], state["previous_cycle"])

    if review and review.rollback_files:
        print("\n=== FOREMAN ROLLING BACK CORRUPTED FILES ===")
        for path in review.rollback_files:
            logical = path.removeprefix("sandbox/")
            if logical in snapshots:
                physical_path = _sandbox_repo_path(logical)
                write_file.func(path=physical_path, content=snapshots[logical])
                print(f"Reverted {physical_path} to pre-batch snapshot")

    print("\n=== FOREMAN GENERATING CORRECTIVE BATCH ===")
    corrective_batch = foreman.create_corrective_batch(
        batch=batch,
        review=review,
        task_results=state["task_results"],
        ledger_context=ledger_ctx,
    )

    corrective_batch, updated_snapshots = _prepare_batch_and_workspace(
        corrective_batch, snapshots
    )

    updated_ledger = record_ledger_entry(
        state["ledger"],
        cycle_id=state["cycle_id"],
        layer="foreman",
        action="corrective_batch",
        target=corrective_batch.batch_id,
        status="warning",
        detail=f"Created corrective batch with tasks: {', '.join(t.task_id for t in corrective_batch.tasks)}",
    )

    return {
        "task_batches": state["task_batches"] + [corrective_batch],
        "file_snapshots": updated_snapshots,
        "rebuild_attempts": state["rebuild_attempts"] + 1,
        "ledger": updated_ledger,
        "phase": "corrective_task_ready",
    }


# ============================================================================
# 7. Planner — Acceptance Review
# ============================================================================


def planner_review_node(state: WorkflowState) -> dict:
    """Planner performs the final evaluation against the initial blueprint."""
    planner = build_planner()
    plan = state["plan"]
    repository_context = _gather_full_repository_context()
    ledger_ctx = format_ledger_for_prompt(state["ledger"], state["previous_cycle"])

    print("\n=== PLANNER FINAL ACCEPTANCE REVIEW ===")
    review = planner.review(
        project_goal=state["project_goal"],
        plan=plan,
        repository_context=repository_context,
        validation_result=state.get("validation_result"),
        ledger_context=ledger_ctx,
    )

    print(review.model_dump_json(indent=2))

    updated_ledger = record_ledger_entry(
        state["ledger"],
        cycle_id=state["cycle_id"],
        layer="planner",
        action="final_review",
        target="project",
        status="success" if review.accepted else "failure",
        detail=review.summary,
    )

    return {
        "planner_review": review,
        "ledger": updated_ledger,
        "phase": "planner_accepted" if review.accepted else "planner_revision_needed",
    }


# ============================================================================
# Routing & Graph Construction
# ============================================================================


def route_after_foreman_review(state: WorkflowState) -> str:
    review = state["foreman_review"]
    if review is None:
        raise RuntimeError("Foreman review missing.")

    validation = state.get("validation_result")
    has_hard_errors = bool(
        validation
        and any(
            err.startswith(("syntax error", "empty file", "missing file", "execution failed"))
            for err in validation.errors
        )
    )

    if review.decision == "accept":
        if not has_hard_errors:
            return "accepted"
        print(
            "WARNING: Foreman chose 'accept' while deterministic execution/syntax errors remain. "
            "Overriding to rework."
        )
        review.accepted = False
        if review.direct_repairs and state["foreman_repair_attempts"] < MAX_FOREMAN_REPAIRS:
            return "self_repair"
        if state["rebuild_attempts"] < MAX_WORKER_REBUILDS:
            return "rebuild"
        return "accepted"

    if review.decision == "repair":
        if state["foreman_repair_attempts"] < MAX_FOREMAN_REPAIRS:
            return "self_repair"
        print("Foreman self-repair attempts exceeded limit. Escalating to rebuild.")

    if state["rebuild_attempts"] >= MAX_WORKER_REBUILDS:
        print(
            f"Worker rebuild attempts reached limit ({MAX_WORKER_REBUILDS}). "
            "Escalating to Planner review."
        )
        return "accepted"

    return "rebuild"


def build_graph():
    graph = StateGraph(WorkflowState)

    graph.add_node("planner", planner_node)
    graph.add_node("foreman", foreman_node)
    graph.add_node("worker", worker_node)
    graph.add_node("foreman_assemble", foreman_assemble_node)
    graph.add_node("validator", validator_node)
    graph.add_node("foreman_review", foreman_review_node)
    graph.add_node("foreman_self_repair", foreman_self_repair_node)
    graph.add_node("foreman_corrective", foreman_corrective_node)
    graph.add_node("planner_review", planner_review_node)

    graph.set_entry_point("planner")

    graph.add_edge("planner", "foreman")
    graph.add_edge("foreman", "worker")
    graph.add_edge("worker", "foreman_assemble")
    graph.add_edge("foreman_assemble", "validator")
    graph.add_edge("validator", "foreman_review")

    graph.add_conditional_edges(
        "foreman_review",
        route_after_foreman_review,
        {
            "self_repair": "foreman_self_repair",
            "rebuild": "foreman_corrective",
            "accepted": "planner_review",
        },
    )

    graph.add_edge("foreman_self_repair", "validator")
    graph.add_edge("foreman_corrective", "worker")
    graph.add_edge("planner_review", END)

    return graph.compile()


app = build_graph()


def run_cycle(
    *,
    cycle_id: int,
    goal: str,
    architecture: str,
    worker_profile: WorkerProfile,
    previous_cycle: CycleResult | None,
) -> CycleResult:
    """Execute a single orchestration cycle."""
    if STAGING_DIR.exists():
        for f in STAGING_DIR.rglob("*.py"):
            f.unlink(missing_ok=True)

    final_state = app.invoke(
        {
            "cycle_id": cycle_id,
            "project_goal": goal,
            "architecture_context": architecture,
            "worker_profile": worker_profile,
            "previous_cycle": previous_cycle,
            "plan": None,
            "task_batches": [],
            "task_results": [],
            "ledger": [],
            "staged_files": {},
            "file_snapshots": {},
            "worker_attempts": {},
            "foreman_repair_attempts": 0,
            "rebuild_attempts": 0,
            "validation_result": None,
            "foreman_review": None,
            "planner_review": None,
            "rework_mode": "none",
            "phase": "initial",
        }
    )

    foreman_review = final_state.get("foreman_review")
    planner_review = final_state.get("planner_review")

    success = (
        planner_review is not None
        and planner_review.accepted
        and (foreman_review is None or foreman_review.accepted)
    )

    outcome = "overall_success" if success else "partial"

    return CycleResult(
        cycle_id=cycle_id,
        plan=final_state["plan"],
        task_batches=final_state["task_batches"],
        task_results=final_state["task_results"],
        ledger=final_state["ledger"],
        planner_review=planner_review,
        outcome=outcome,
    )


def run_project(
    *,
    goal: str,
    architecture: str,
    worker_profile: WorkerProfile,
    max_cycles: int = 3,
) -> list[CycleResult]:
    """Execute the outer Planner loop across multiple cycles."""
    history: list[CycleResult] = []
    previous_cycle: CycleResult | None = None

    for cycle_id in range(1, max_cycles + 1):
        print(f"\n==========================================")
        print(f"=== STARTING ORCHESTRATION CYCLE {cycle_id}/{max_cycles} ===")
        print(f"==========================================")

        result = run_cycle(
            cycle_id=cycle_id,
            goal=goal,
            architecture=architecture,
            worker_profile=worker_profile,
            previous_cycle=previous_cycle,
        )
        history.append(result)

        if result.outcome == "overall_success":
            print(f"\n=== PROJECT ACCEPTED IN CYCLE {cycle_id} ===")
            break

        print(f"\n=== CYCLE {cycle_id} INCOMPLETE ({result.outcome}) — RE-PLANNING ===")
        previous_cycle = result

    return history

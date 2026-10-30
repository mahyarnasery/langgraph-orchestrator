import argparse
import os
import shutil
import sys

from src.harnesses.aider import list_workspace_files
from src.orchestration.checkpoint import (
    clear_checkpoint,
    format_checkpoint_status,
    load_checkpoint,
)
from src.orchestration.graph import (
    _build_effective_validation_batch,
    configure_loop_limits,
    run_project,
    run_staged_project,
)
from src.orchestration.models import configure_models, get_worker_model
from src.orchestration.schemas import Task, TaskBatch, TaskResult, WorkerProfile
from src.orchestration.validator import emit_pytest_suite, validate_batch
from src.orchestration.workspace import get_workspace_root, set_active_workspace


def clean_active_workspace() -> None:
    ws_root = get_workspace_root()
    ws_root.mkdir(parents=True, exist_ok=True)
    for fpath in list_workspace_files():
        fpath.unlink(missing_ok=True)
    for child in sorted(ws_root.iterdir()):
        if child.is_dir() and child.name != ".git":
            shutil.rmtree(child, ignore_errors=True)
    clear_checkpoint()


def run_validate_only(emit_tests: bool = False) -> int:
    """Run deterministic validation (and optional pytest emission) on the active workspace without LLMs."""
    ws_root = get_workspace_root()
    ckpt = load_checkpoint()

    if ckpt is not None and ckpt.task_batches:
        batch = _build_effective_validation_batch(ckpt.task_batches)
        task_results = list(ckpt.task_results)
    else:
        rel_files = [
            str(p.relative_to(ws_root))
            for p in list_workspace_files()
        ]
        if not rel_files:
            print(f"No files found to validate in {ws_root}")
            return 1
        batch = TaskBatch(
            batch_id="manual_validate_only",
            objective=f"Validate workspace {ws_root}",
            tasks=[
                Task(
                    task_id=f"file_{idx}",
                    objective=f"Validate {rf}",
                    target_symbol=None,
                    scope=rf,
                    relevant_files=[rf],
                    acceptance_criteria=[],
                )
                for idx, rf in enumerate(rel_files, start=1)
            ],
            relevant_files=rel_files,
            execution_order=[f"file_{idx}" for idx in range(1, len(rel_files) + 1)],
        )
        task_results = [
            TaskResult(task_id=t.task_id, status="completed", summary="On disk", files_changed=t.relevant_files)
            for t in batch.tasks
        ]

    val = validate_batch(batch, task_results)
    print(f"\n=== VALIDATE-ONLY RESULT ({ws_root}) ===")
    print(val.model_dump_json(indent=2))

    if val.passed and emit_tests:
        emit_pytest_suite(batch)

    return 0 if val.passed else 1


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the 3-layer local LangGraph AI coding orchestrator."
    )
    parser.add_argument(
        "goal",
        nargs="?",
        default="",
        help="High-level project goal or coding instruction (optional with --resume, --status, or --validate-only).",
    )
    parser.add_argument(
        "-w",
        "--workspace",
        default=None,
        help="Target workspace directory to build/inspect/resume in (default: sandbox/).",
    )
    parser.add_argument(
        "--arch",
        default="Modular Python workspace",
        help="Architecture context and constraints.",
    )
    parser.add_argument(
        "--cycles",
        type=int,
        default=3,
        help="Maximum outer Planner cycles per stage/project (default: 3).",
    )
    parser.add_argument(
        "--staged",
        action="store_true",
        help="Decompose the project goal into sequential multi-stage milestones.",
    )
    parser.add_argument(
        "--max-stages",
        type=int,
        default=3,
        help="Maximum stages when decomposing with --staged (default: 3).",
    )
    parser.add_argument(
        "--stop-after-stage",
        type=int,
        default=None,
        help="Stop cleanly after completing stage N in --staged mode.",
    )
    parser.add_argument(
        "--stop-after-tasks",
        type=int,
        default=None,
        help="Stop and checkpoint after executing N Worker Aider tasks in this session.",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Inspect the current workspace checkpoint and disk reconciliation status without invoking LLMs.",
    )
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="Run only the Planner blueprint (and stage decomposition if --staged), save to checkpoint, and exit.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Run deterministic validation (syntax, execution, unit assertions, pytest) on the workspace without LLMs.",
    )
    parser.add_argument(
        "--emit-tests",
        action="store_true",
        help="Export passing unit assertions into <workspace>/tests/test_*.py after validation succeeds.",
    )
    parser.add_argument(
        "--planner-model",
        default=None,
        help="Override Planner Ollama model name (default: gemma4:e4b).",
    )
    parser.add_argument(
        "--foreman-model",
        default=None,
        help="Override Foreman Ollama model name (default: ibm/granite4.1:8b).",
    )
    parser.add_argument(
        "--worker-model",
        default=None,
        help="Override Worker Ollama model name (default: ibm/granite4.1:3b).",
    )
    parser.add_argument(
        "--max-repairs",
        type=int,
        default=None,
        help="Override MAX_FOREMAN_REPAIRS (default: 2).",
    )
    parser.add_argument(
        "--max-rebuilds",
        type=int,
        default=None,
        help="Override MAX_WORKER_REBUILDS (default: 3).",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Clean existing files, subdirectories, and checkpoints in the target workspace before starting.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from the last saved checkpoint in <workspace>/.checkpoint/state.json.",
    )

    args = parser.parse_args()

    ws_root = set_active_workspace(args.workspace)
    configure_models(
        planner_model=args.planner_model,
        foreman_model=args.foreman_model,
        worker_model=args.worker_model,
    )
    configure_loop_limits(
        max_repairs=args.max_repairs,
        max_rebuilds=args.max_rebuilds,
    )

    if args.clean:
        clean_active_workspace()
        print(f"Cleaned workspace {ws_root} and checkpoints.")

    if args.status:
        print(format_checkpoint_status())
        sys.exit(0)

    if args.validate_only:
        sys.exit(run_validate_only(emit_tests=args.emit_tests))

    goal = args.goal.strip()
    arch = args.arch
    ckpt = load_checkpoint() if args.resume else None
    if args.resume and not goal:
        if ckpt is None or not ckpt.project_goal.strip():
            parser.error(
                f"--resume was specified without a goal, and no saved checkpoint was found in {ws_root}/.checkpoint/state.json."
            )
        goal = ckpt.project_goal
        arch = ckpt.architecture_context or arch
    elif not goal:
        parser.error(
            "A project goal is required unless using --resume, --status, or --validate-only."
        )

    profile = WorkerProfile(
        model=get_worker_model(),
        context_limit="limited",
        strengths=["localized single-file edits", "writing clean functions, methods, and config files"],
        avoid=["architecture redesign", "interactive input() prompts"],
        execution_mode="one_session_per_task",
    )

    if args.stop_after_tasks is not None and args.stop_after_tasks > 0:
        os.environ["ORCHESTRATOR_STOP_AFTER_TASKS"] = str(args.stop_after_tasks)

    try:
        use_staged = args.staged or bool(ckpt and ckpt.stages)
        if use_staged:
            history = run_staged_project(
                goal=goal,
                architecture=arch,
                worker_profile=profile,
                max_stages=args.max_stages,
                max_cycles_per_stage=args.cycles,
                resume=args.resume,
                workspace=ws_root,
                plan_only=args.plan_only,
                stop_after_stage=args.stop_after_stage,
                emit_tests=args.emit_tests,
            )
        else:
            history = run_project(
                goal=goal,
                architecture=arch,
                worker_profile=profile,
                max_cycles=args.cycles,
                resume=args.resume,
                workspace=ws_root,
                plan_only=args.plan_only,
                emit_tests=args.emit_tests,
            )
    except KeyboardInterrupt as exc:
        print(f"\n[INTERRUPTED] {exc}")
        print(f"Checkpoint saved at {ws_root}/.checkpoint/state.json. Re-run with --resume to continue.")
        sys.exit(130)
    finally:
        os.environ.pop("ORCHESTRATOR_STOP_AFTER_TASKS", None)

    final = history[-1]
    print(f"\n=== FINAL OUTCOME ({len(history)} cycle(s)) in {ws_root}: {final.outcome} ===")
    if final.planner_review:
        print(f"Summary: {final.planner_review.summary}")

    sys.exit(0 if final.outcome in ("overall_success", "planned") else 1)


if __name__ == "__main__":
    main()

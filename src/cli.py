import argparse
import shutil
import sys

from src.orchestration.checkpoint import clear_checkpoint, load_checkpoint
from src.orchestration.graph import WORKER_WORKSPACE, run_project
from src.orchestration.schemas import WorkerProfile


def clean_sandbox_workspace() -> None:
    WORKER_WORKSPACE.mkdir(parents=True, exist_ok=True)
    for py_file in WORKER_WORKSPACE.rglob("*.py"):
        py_file.unlink(missing_ok=True)
    for child in sorted(WORKER_WORKSPACE.iterdir()):
        if child.is_dir() and child.name != ".git":
            shutil.rmtree(child, ignore_errors=True)
    clear_checkpoint()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the 3-layer local LangGraph AI coding orchestrator."
    )
    parser.add_argument(
        "goal",
        nargs="?",
        default="",
        help="High-level project goal or coding instruction (optional when using --resume).",
    )
    parser.add_argument(
        "--arch",
        default="Modular Python workspace in sandbox/",
        help="Architecture context and constraints.",
    )
    parser.add_argument(
        "--cycles",
        type=int,
        default=3,
        help="Maximum outer Planner cycles (default: 3).",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Clean existing .py files, subdirectories, and checkpoints in sandbox/ before starting.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from the last saved checkpoint in sandbox/.checkpoint/state.json.",
    )

    args = parser.parse_args()

    WORKER_WORKSPACE.mkdir(parents=True, exist_ok=True)
    if args.clean:
        clean_sandbox_workspace()
        print("Cleaned sandbox/ workspace and checkpoints.")

    goal = args.goal.strip()
    arch = args.arch
    if args.resume and not goal:
        ckpt = load_checkpoint()
        if ckpt is None or not ckpt.project_goal.strip():
            parser.error("--resume was specified without a goal, and no saved checkpoint was found.")
        goal = ckpt.project_goal
        arch = ckpt.architecture_context or arch
    elif not goal:
        parser.error("A project goal is required unless resuming from an existing checkpoint (--resume).")

    profile = WorkerProfile(
        model="ibm/granite4.1:3b",
        context_limit="limited",
        strengths=["localized single-file edits", "writing clean functions"],
        avoid=["architecture redesign", "interactive input() prompts"],
        execution_mode="one_session_per_task",
    )

    history = run_project(
        goal=goal,
        architecture=arch,
        worker_profile=profile,
        max_cycles=args.cycles,
        resume=args.resume,
    )

    final = history[-1]
    print(f"\n=== FINAL OUTCOME ({len(history)} cycle(s)): {final.outcome} ===")
    if final.planner_review:
        print(f"Summary: {final.planner_review.summary}")

    sys.exit(0 if final.outcome == "overall_success" else 1)


if __name__ == "__main__":
    main()

import argparse
import shutil
import sys

from src.orchestration.checkpoint import clear_checkpoint, load_checkpoint
from src.orchestration.graph import run_project, run_staged_project
from src.orchestration.schemas import WorkerProfile
from src.orchestration.workspace import get_workspace_root, set_active_workspace


def clean_active_workspace() -> None:
    ws_root = get_workspace_root()
    ws_root.mkdir(parents=True, exist_ok=True)
    for py_file in ws_root.rglob("*.py"):
        py_file.unlink(missing_ok=True)
    for child in sorted(ws_root.iterdir()):
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
        "-w",
        "--workspace",
        default=None,
        help="Target workspace directory to build/resume the project in (default: sandbox/).",
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
        help="Maximum outer Planner cycles (default: 3).",
    )
    parser.add_argument(
        "--staged",
        action="store_true",
        help="Decompose the project goal into sequential multi-stage milestones.",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Clean existing .py files, subdirectories, and checkpoints in the target workspace before starting.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from the last saved checkpoint in <workspace>/.checkpoint/state.json.",
    )

    args = parser.parse_args()

    ws_root = set_active_workspace(args.workspace)
    if args.clean:
        clean_active_workspace()
        print(f"Cleaned workspace {ws_root} and checkpoints.")

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
        parser.error("A project goal is required unless resuming from an existing checkpoint (--resume).")

    profile = WorkerProfile(
        model="ibm/granite4.1:3b",
        context_limit="limited",
        strengths=["localized single-file edits", "writing clean functions and methods"],
        avoid=["architecture redesign", "interactive input() prompts"],
        execution_mode="one_session_per_task",
    )

    use_staged = args.staged or bool(ckpt and ckpt.stages)
    if use_staged:
        history = run_staged_project(
            goal=goal,
            architecture=arch,
            worker_profile=profile,
            max_cycles_per_stage=args.cycles,
            resume=args.resume,
            workspace=ws_root,
        )
    else:
        history = run_project(
            goal=goal,
            architecture=arch,
            worker_profile=profile,
            max_cycles=args.cycles,
            resume=args.resume,
            workspace=ws_root,
        )

    final = history[-1]
    print(f"\n=== FINAL OUTCOME ({len(history)} cycle(s)) in {ws_root}: {final.outcome} ===")
    if final.planner_review:
        print(f"Summary: {final.planner_review.summary}")

    sys.exit(0 if final.outcome == "overall_success" else 1)


if __name__ == "__main__":
    main()

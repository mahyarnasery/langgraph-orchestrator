import argparse
from pathlib import Path
import sys

from src.orchestration.graph import WORKER_WORKSPACE, run_project
from src.orchestration.schemas import WorkerProfile


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the 3-layer local LangGraph AI coding orchestrator."
    )
    parser.add_argument(
        "goal",
        help="High-level project goal or coding instruction.",
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
        help="Clean existing .py files in sandbox/ before starting.",
    )

    args = parser.parse_args()

    WORKER_WORKSPACE.mkdir(parents=True, exist_ok=True)
    if args.clean:
        for py_file in WORKER_WORKSPACE.rglob("*.py"):
            py_file.unlink(missing_ok=True)
        print("Cleaned sandbox/ workspace.")

    profile = WorkerProfile(
        model="ibm/granite4.1:3b",
        context_limit="limited",
        strengths=["localized single-file edits", "writing clean functions"],
        avoid=["architecture redesign", "interactive input() prompts"],
        execution_mode="one_session_per_task",
    )

    history = run_project(
        goal=args.goal,
        architecture=args.arch,
        worker_profile=profile,
        max_cycles=args.cycles,
    )

    final = history[-1]
    print(f"\n=== FINAL OUTCOME ({len(history)} cycle(s)): {final.outcome} ===")
    if final.planner_review:
        print(f"Summary: {final.planner_review.summary}")

    sys.exit(0 if final.outcome == "overall_success" else 1)


if __name__ == "__main__":
    main()

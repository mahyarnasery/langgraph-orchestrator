from pathlib import Path
import subprocess
import sys

from src.orchestration.graph import run_cycle
from src.orchestration.schemas import WorkerProfile


REPO_ROOT = Path("/mnt/ai/langgraph-orchestrator").resolve()
WORKSPACE_ROOT = (REPO_ROOT / "sandbox").resolve()

TARGET = WORKSPACE_ROOT / "hello.py"
ROOT_TARGET = REPO_ROOT / "hello.py"


def main() -> None:
    # --------------------------------------------------------
    # Clean previous smoke-test artifacts.
    # --------------------------------------------------------

    TARGET.unlink(missing_ok=True)

    # Remove the accidental file produced by the previous
    # incorrectly isolated Aider invocation.
    ROOT_TARGET.unlink(missing_ok=True)

    profile = WorkerProfile(
        model="ibm/granite4.1:3b",
        context_limit="limited",
        strengths=["localized edits"],
        avoid=["architecture redesign"],
        execution_mode="aider",
    )

    result = run_cycle(
        cycle_id=1,
        goal="""
            Create hello.py.

            The file must contain exactly:

            print("Hello Alpha04")

            Do not add functions, comments, wrappers, or extra code.
            """,
        architecture="Minimal smoke test",
        worker_profile=profile,
        previous_cycle=None,
    )

    print(result.model_dump_json(indent=2))

    # --------------------------------------------------------
    # Alpha04 isolation assertions.
    # --------------------------------------------------------

    if not TARGET.is_file():
        raise AssertionError(
            f"Expected Worker file was not created: {TARGET}"
        )

    if ROOT_TARGET.exists():
        raise AssertionError(
            "Worker escaped the sandbox and created "
            f"a file in the repository root: {ROOT_TARGET}"
        )

    # --------------------------------------------------------
    # Debug: verify file content before subprocess execution.
    # --------------------------------------------------------
    print(f"\n=== DEBUG: Checking file at {TARGET} ===")
    print(f"File exists: {TARGET.is_file()}")
    print(f"File size: {TARGET.stat().st_size} bytes")
    content = TARGET.read_text()
    print(f"File content (repr): {content!r}")
    print(f"File content (raw):\n{content}")

    # --------------------------------------------------------
    # Verify actual generated behavior.
    # --------------------------------------------------------

    completed = subprocess.run(
        [
            sys.executable,
            str(TARGET),
        ],
        cwd=WORKSPACE_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    print(f"\n=== DEBUG: Subprocess result ===")
    print(f"Return code: {completed.returncode}")
    print(f"Stdout (repr): {completed.stdout!r}")
    print(f"Stderr (repr): {completed.stderr!r}")

    if completed.returncode != 0:
        raise AssertionError(
            "hello.py failed to execute:\n"
            f"{completed.stderr}"
        )

    if completed.stdout.strip() != "Hello Alpha04":
        raise AssertionError(
            "Unexpected hello.py output: "
            f"{completed.stdout!r}"
        )

    print("\n=== ALPHA04 SMOKE TEST PASSED ===")
    print(f"Worker file: {TARGET}")
    print(f"Output: {completed.stdout.strip()}")


if __name__ == "__main__":
    main()

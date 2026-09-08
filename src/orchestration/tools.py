from pathlib import Path

from langchain_core.tools import tool


REPO_ROOT = Path("/mnt/ai/langgraph-orchestrator").resolve()
WORKSPACE_ROOT = (REPO_ROOT / "sandbox").resolve()


def _safe_path(path: str) -> Path:
    """
    Resolve a relative path and ensure it remains inside the
    orchestration workspace.

    Absolute paths and path traversal outside WORKSPACE_ROOT
    are rejected.
    """
    candidate = Path(path)

    if candidate.is_absolute():
        raise ValueError(
            f"Absolute paths are not allowed: {path}"
        )

    # Normalize sandbox-prefixed paths if provided
    normalized = candidate
    if str(candidate).startswith("sandbox/"):
        normalized = Path(str(candidate).removeprefix("sandbox/"))

    resolved = (WORKSPACE_ROOT / normalized).resolve()

    try:
        resolved.relative_to(WORKSPACE_ROOT)
    except ValueError as exc:
        raise ValueError(
            f"Path escapes orchestration workspace: {path}"
        ) from exc
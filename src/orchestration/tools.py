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

    return resolved


@tool
def read_file(path: str) -> dict:
    """
    Read content of a file within the workspace.
    """
    file_path = _safe_path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    if not file_path.is_file():
        raise ValueError(f"Path is not a file: {path}")

    content = file_path.read_text(encoding="utf-8")
    return {"path": path, "content": content}


@tool
def create_file(path: str, content: str = "") -> dict:
    """
    Create a new file within the workspace with optional initial content.
    Creates parent directories if necessary.
    """
    file_path = _safe_path(path)
    if file_path.exists():
        raise FileExistsError(f"File already exists: {path}")

    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(content, encoding="utf-8")
    return {"path": path, "status": "created"}
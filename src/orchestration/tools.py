from pathlib import Path

from langchain_core.tools import tool


REPO_ROOT = Path("/mnt/ai/langgraph-orchestrator").resolve()
WORKSPACE_ROOT = (REPO_ROOT / "sandbox").resolve()


def _safe_path(path: str) -> Path:
    """
    Resolve a relative path and ensure it remains inside the
    orchestration workspace.
    """
    candidate = Path(path)

    if candidate.is_absolute():
        raise ValueError(f"Absolute paths are not allowed: {path}")

    normalized = candidate
    if str(candidate).startswith("sandbox/"):
        normalized = Path(str(candidate).removeprefix("sandbox/"))

    resolved = (WORKSPACE_ROOT / normalized).resolve()

    try:
        resolved.relative_to(WORKSPACE_ROOT)
    except ValueError as exc:
        raise ValueError(f"Path escapes orchestration workspace: {path}") from exc

    return resolved


def _ensure_trailing_newline(content: str) -> str:
    """Ensure non-empty file content ends with a newline to prevent line concatenation."""
    if content and not content.endswith("\n"):
        return content + "\n"
    return content


@tool
def read_file(path: str) -> dict:
    """Read content of a file within the workspace."""
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
    file_path.write_text(_ensure_trailing_newline(content), encoding="utf-8")
    return {"path": path, "status": "created"}


@tool
def write_file(path: str, content: str) -> dict:
    """Overwrite or write content to a file within the workspace."""
    file_path = _safe_path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(_ensure_trailing_newline(content), encoding="utf-8")
    return {"path": path, "status": "written"}


@tool
def edit_file(path: str, old_text: str, new_text: str) -> dict:
    """Replace target text in a file within the workspace."""
    file_path = _safe_path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    current_content = file_path.read_text(encoding="utf-8")
    if old_text not in current_content:
        raise ValueError(f"Target text not found in {path}")

    updated_content = current_content.replace(old_text, new_text, 1)
    file_path.write_text(_ensure_trailing_newline(updated_content), encoding="utf-8")
    return {"path": path, "status": "edited"}

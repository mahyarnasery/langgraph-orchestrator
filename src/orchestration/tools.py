from pathlib import Path

from langchain_core.tools import tool

from .workspace import (
    check_read_permission,
    get_workspace_root,
    normalize_workspace_rel_path,
    request_external_write_permission,
)


def _resolve_read_path(path: str) -> Path:
    """
    Resolve a file path for reading, enforcing the active AccessMode:
    - Mode 1 ('sandbox_only'): Must be inside `sandbox/`.
    - Mode 2 ('read_external', Default) & Mode 3 ('allow_external'): Can read inside `sandbox/`
      or an external path outside `sandbox/`.
    """
    workspace_root = get_workspace_root()
    candidate = Path(path).expanduser()

    if candidate.is_absolute():
        resolved = candidate.resolve()
    else:
        normalized = Path(normalize_workspace_rel_path(path))
        ws_candidate = (workspace_root / normalized).resolve()
        if ws_candidate.exists() or ws_candidate.is_relative_to(workspace_root):
            resolved = ws_candidate
        else:
            resolved = (Path.cwd() / candidate).resolve()

    return check_read_permission(resolved)


def _safe_write_path(path: str) -> Path:
    """
    Resolve a path for creating/modifying a file and enforce both workspace
    containment and the active AccessMode (Mode 1/2: strictly `sandbox/` only;
    Mode 3: prompts for confirmation if outside `sandbox/`).
    """
    workspace_root = get_workspace_root()
    candidate = Path(path)

    if candidate.is_absolute():
        raise ValueError(f"Absolute paths are not allowed for writes: {path}")

    normalized = Path(normalize_workspace_rel_path(path))
    resolved = (workspace_root / normalized).resolve()

    try:
        resolved.relative_to(workspace_root)
    except ValueError as exc:
        raise ValueError(f"Path escapes orchestration workspace: {path}") from exc

    request_external_write_permission(resolved, operation="write/modify file")
    return resolved


def _ensure_trailing_newline(content: str) -> str:
    """Ensure non-empty file content ends with a newline to prevent line concatenation."""
    if content and not content.endswith("\n"):
        return content + "\n"
    return content


@tool
def read_file(path: str) -> dict:
    """Read content of a file (respecting the active sandbox/external read access mode)."""
    file_path = _resolve_read_path(path)
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
    file_path = _safe_write_path(path)
    if file_path.exists():
        raise FileExistsError(f"File already exists: {path}")

    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(_ensure_trailing_newline(content), encoding="utf-8")
    return {"path": path, "status": "created"}


@tool
def write_file(path: str, content: str) -> dict:
    """Overwrite or write content to a file within the workspace."""
    file_path = _safe_write_path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(_ensure_trailing_newline(content), encoding="utf-8")
    return {"path": path, "status": "written"}


@tool
def edit_file(path: str, old_text: str, new_text: str) -> dict:
    """Replace target text in a file within the workspace."""
    file_path = _safe_write_path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    current_content = file_path.read_text(encoding="utf-8")
    if old_text not in current_content:
        raise ValueError(f"Target text not found in {path}")

    updated_content = current_content.replace(old_text, new_text, 1)
    file_path.write_text(_ensure_trailing_newline(updated_content), encoding="utf-8")
    return {"path": path, "status": "edited"}

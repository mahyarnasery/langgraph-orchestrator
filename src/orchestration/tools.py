from pathlib import Path
import re

from langchain_core.tools import tool

from .workspace import (
    check_read_permission,
    get_workspace_root,
    normalize_workspace_rel_path,
    request_external_write_permission,
)


def _resolve_read_path(path: str) -> Path:
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


def sanitize_file_content(path: str, content: str) -> str:
    """
    Strip accidental outer Markdown code fences (```python ... ``` or ~~~json ... ~~~)
    and stray terminal progress bars when writing code, config, or data files.
    """
    if not content:
        return ""

    suffix = Path(path).suffix.lower()
    if suffix == ".md":
        return content if content.endswith("\n") else content + "\n"

    cleaned = content.strip()
    fence_match = re.match(
        r"^(?:```|~~~)[a-zA-Z0-9_-]*\s*\n(.*?)\n?(?:```|~~~)\s*$",
        cleaned,
        re.DOTALL,
    )
    if fence_match:
        cleaned = fence_match.group(1).strip()
    else:
        lines = cleaned.splitlines()
        if lines and lines[0].strip().startswith(("```", "~~~")):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith(("```", "~~~")):
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()

    # Remove accidental terminal progress-bar lines (e.g. '11 / 11 lines [████] 100%')
    filtered_lines = [
        ln for ln in cleaned.splitlines()
        if not re.search(r"\d+\s*/\s*\d+\s+lines\s+\[", ln)
    ]
    cleaned = "\n".join(filtered_lines).strip()
    return (cleaned + "\n") if cleaned else ""


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
    file_path.write_text(sanitize_file_content(path, content), encoding="utf-8")
    return {"path": path, "status": "created"}


@tool
def write_file(path: str, content: str) -> dict:
    """Overwrite or write content to a file within the workspace."""
    file_path = _safe_write_path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(sanitize_file_content(path, content), encoding="utf-8")
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
    file_path.write_text(sanitize_file_content(path, updated_content), encoding="utf-8")
    return {"path": path, "status": "edited"}

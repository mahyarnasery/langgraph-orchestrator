from pathlib import Path

from langchain_core.tools import tool


REPO_ROOT = Path("/mnt/ai/langgraph-orchestrator").resolve()
WORKSPACE_ROOT = REPO_ROOT


def _safe_path(path: str) -> Path:
    candidate = Path(path)

    if candidate.is_absolute():
        raise ValueError(
            f"Absolute paths are not allowed: {path}"
        )

    resolved = (WORKSPACE_ROOT / candidate).resolve()

    try:
        resolved.relative_to(WORKSPACE_ROOT)
    except ValueError as exc:
        raise ValueError(
            f"Path escapes orchestration workspace: {path}"
        ) from exc

    return resolved

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

    resolved = (REPO_ROOT / candidate).resolve()

    try:
        resolved.relative_to(WORKSPACE_ROOT)
    except ValueError as exc:
        raise ValueError(
            f"Path escapes worker workspace: {path}"
        ) from exc

    return resolved


def _ensure_workspace() -> None:
    """Ensure the isolated Worker workspace exists."""

    WORKSPACE_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )


@tool
def read_file(path: str) -> dict:
    """
    Read an existing text file inside the orchestration workspace.

    Available to both the Foreman and Worker.

    This tool never creates or modifies files.
    """

    _ensure_workspace()

    target = _safe_path(path)

    if not target.is_file():
        raise FileNotFoundError(
            f"File does not exist: {path}"
        )

    print(
        f"\n*** TOOL: read_file({path}) ***"
    )

    return {
        "tool": "read_file",
        "path": path,
        "result": "OK",
        "content": target.read_text(encoding="utf-8"),
    }


@tool
def create_file(path: str, content: str = "") -> dict:
    """
    Create a new text file inside the orchestration workspace.

    This tool is available only to the Foreman / Integrator.

    The target file must not already exist.

    Parent directories may be created automatically, but only
    when the resulting path remains inside WORKSPACE_ROOT.
    """

    _ensure_workspace()

    target = _safe_path(path)

    
    if target.exists():
        raise FileExistsError(
            f"File already exists: {path}"
        )

    if target.parent.exists() and not target.parent.is_dir():
        raise NotADirectoryError(
            f"Parent path is not a directory: {target.parent}"
        )

    target.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    target.write_text(
        content,
        encoding="utf-8",
    )

    print(
        f"\n*** FOREMAN TOOL: create_file({path}) ***"
    )

    return {
        "tool": "create_file",
        "path": path,
        "result": "FILE_CREATED",
    }


@tool
def edit_file(
    path: str,
    old_text: str,
    new_text: str,
) -> dict:
    """
    Apply one deterministic edit to an existing file.

    Modes
    -----
    1. Initial implementation:
       - Target file is empty/whitespace only.
       - Entire file becomes new_text.

    2. Replacement:
       - old_text must exist exactly once.
       - Exactly one occurrence is replaced.

    Safety
    ------
    - Never reports success if nothing actually changed.
    - Rejects ambiguous matches.
    - Leaves the file untouched on failure.

    If the file is structurally broken, use replace_file to
    rewrite the complete corrected file.
    """

    _ensure_workspace()

    target = _safe_path(path)

    if not target.is_file():
        raise FileNotFoundError(
            f"File does not exist: {path}"
        )

    print(
        f"\n*** TOOL: edit_file({path}) ***"
    )

    content = target.read_text(encoding="utf-8")

    # --------------------------------------------------------
    # Initial implementation mode
    # --------------------------------------------------------
    if content.strip() == "":
        if content == new_text:
            return {
                "tool": "edit_file",
                "path": path,
                "result": "NO_CHANGE",
            }

        target.write_text(
            new_text,
            encoding="utf-8",
        )

        return {
            "tool": "edit_file",
            "path": path,
            "result": "EDIT_SUCCESS",
        }

    # --------------------------------------------------------
    # Replacement mode
    # --------------------------------------------------------
    if old_text not in content:
        return {
            "tool": "edit_file",
            "path": path,
            "result": "OLD_TEXT_NOT_FOUND",
        }

    if content.count(old_text) > 1:
        return {
            "tool": "edit_file",
            "path": path,
            "result": "AMBIGUOUS_EDIT",
        }

    updated = content.replace(
        old_text,
        new_text,
        1,
    )

    if updated == content:
        return {
            "tool": "edit_file",
            "path": path,
            "result": "NO_CHANGE",
        }

    target.write_text(
        updated,
        encoding="utf-8",
    )

    return {
        "tool": "edit_file",
        "path": path,
        "result": "EDIT_SUCCESS",
    }


@tool
def replace_file(
    path: str,
    content: str,
) -> dict:
    """
    Replace the entire contents of an existing file.

    Use this only after reading the current file.
    It is intended for repairing malformed files where a
    precise substring replacement is unreliable.
    """

    _ensure_workspace()

    target = _safe_path(path)

    if not target.is_file():
        raise FileNotFoundError(
            f"File does not exist: {path}"
        )

    print(
        f"\n*** TOOL: replace_file({path}) ***"
    )

    current = target.read_text(
        encoding="utf-8"
    )

    if current == content:
        return {
            "tool": "replace_file",
            "path": path,
            "result": "NO_CHANGE",
        }

    target.write_text(
        content,
        encoding="utf-8",
    )

    return {
        "tool": "replace_file",
        "path": path,
        "result": "EDIT_SUCCESS",
    }


# ------------------------------------------------------------
# Tool access by orchestration layer
# ------------------------------------------------------------

FOREMAN_TOOLS = [
    read_file,
    create_file,
    edit_file,
    replace_file,
]


WORKER_TOOLS = [
    read_file,
    edit_file,
]
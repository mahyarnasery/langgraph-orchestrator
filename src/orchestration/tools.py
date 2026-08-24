from pathlib import Path

from langchain_core.tools import tool


WORKSPACE_ROOT = (
    Path("/mnt/ai/langgraph-orchestrator/sandbox")
    .resolve()
)


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

    resolved = (WORKSPACE_ROOT / candidate).resolve()

    try:
        resolved.relative_to(WORKSPACE_ROOT)
    except ValueError as exc:
        raise ValueError(
            f"Path escapes worker workspace: {path}"
        ) from exc

    return resolved


@tool
def read_file(path: str) -> dict:
    """
    Read an existing text file inside the orchestration workspace.

    Available to both the Foreman and Worker.

    This tool never creates or modifies files.
    """

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

    target = _safe_path(path)

    if target.exists():
        raise FileExistsError(
            f"File already exists: {path}"
        )

    # Fail clearly if an existing path component prevents the
    # parent directory from being created.
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
    Apply one narrowly scoped text replacement to an existing file.

    Available to both the Foreman and Worker.

    The target file must already exist.

    Exactly one occurrence of old_text must exist.
    If zero or multiple occurrences are found, no changes are made.
    """

    target = _safe_path(path)

    if not target.is_file():
        raise FileNotFoundError(
            f"File does not exist: {path}"
        )

    print(
        f"\n*** TOOL: edit_file({path}) ***"
    )

    content = target.read_text(
        encoding="utf-8"
    )

    if old_text not in content:
        return {
    "tool": "edit_file",
    "path": path,
    "result": "OLD_TEXT_NOT_FOUND",
}

    occurrences = content.count(old_text)

    if occurrences > 1:
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

    target.write_text(
        updated,
        encoding="utf-8",
    )

    return {
        "tool": "edit_file",
        "path": path,
        "result": "EDIT_SUCCESS",
    }


# ------------------------------------------------------------
# Tool access by orchestration layer
# ------------------------------------------------------------

# Foreman / Integrator
#
# The Foreman is the repository-aware layer.
#
# It can:
#   - inspect existing files
#   - create required files
#   - create required directories indirectly through create_file
#   - make repository-level edits
#
# This allows the Foreman to prepare the repository before
# delegating implementation work to the cheap Worker.
FOREMAN_TOOLS = [
    read_file,
    create_file,
    edit_file,
]


# Cheap Worker
#
# The Worker is deliberately restricted.
#
# It can:
#   - inspect existing files
#   - modify existing files through narrowly scoped edits
#
# It CANNOT:
#   - create files
#   - create directories
#   - decide that a new repository file should exist
#
# If the Foreman failed to prepare a required file, the Worker
# must report the problem rather than creating the file itself.
WORKER_TOOLS = [
    read_file,
    edit_file,
]
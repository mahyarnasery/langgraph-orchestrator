from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path("/mnt/ai/langgraph-orchestrator").resolve()
ORCHESTRATOR_SRC = (REPO_ROOT / "src").resolve()
DEFAULT_WORKSPACE_ROOT = (REPO_ROOT / "sandbox").resolve()

_ACTIVE_WORKSPACE_ROOT: Path = DEFAULT_WORKSPACE_ROOT


def set_active_workspace(path: str | Path | None) -> Path:
    """
    Set and initialize the active target workspace directory where Worker/Foreman
    files, `.staging/`, and `.checkpoint/state.json` reside.
    """
    global _ACTIVE_WORKSPACE_ROOT

    if path is None or str(path).strip() == "":
        resolved = DEFAULT_WORKSPACE_ROOT
    else:
        candidate = Path(path).expanduser()
        if not candidate.is_absolute():
            resolved = (Path.cwd() / candidate).resolve()
        else:
            resolved = candidate.resolve()

    if resolved == Path("/") or resolved == REPO_ROOT or resolved.is_relative_to(ORCHESTRATOR_SRC):
        raise ValueError(
            f"Refusing unsafe workspace path '{resolved}': workspace cannot be '/', "
            f"the orchestrator root '{REPO_ROOT}', or inside '{ORCHESTRATOR_SRC}'."
        )

    resolved.mkdir(parents=True, exist_ok=True)
    _ACTIVE_WORKSPACE_ROOT = resolved
    return _ACTIVE_WORKSPACE_ROOT


def get_workspace_root() -> Path:
    """Return the currently active workspace root directory."""
    _ACTIVE_WORKSPACE_ROOT.mkdir(parents=True, exist_ok=True)
    return _ACTIVE_WORKSPACE_ROOT


def get_staging_dir() -> Path:
    """Return `.staging/` inside the currently active workspace."""
    staging = get_workspace_root() / ".staging"
    staging.mkdir(parents=True, exist_ok=True)
    return staging


def get_checkpoint_dir() -> Path:
    """Return `.checkpoint/` inside the currently active workspace."""
    ckpt_dir = get_workspace_root() / ".checkpoint"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    return ckpt_dir


def get_checkpoint_file() -> Path:
    """Return `.checkpoint/state.json` inside the currently active workspace."""
    return get_checkpoint_dir() / "state.json"


def normalize_workspace_rel_path(path: str) -> str:
    """
    Normalize a logical path into a clean POSIX path relative to the active workspace,
    stripping leading `./`, `sandbox/`, or `<active_workspace_name>/` prefixes.
    """
    cleaned = path.strip().replace("\\", "/")
    while cleaned.startswith("./"):
        cleaned = cleaned[2:]
    cleaned = cleaned.removeprefix("sandbox/").strip("/")
    ws_name = _ACTIVE_WORKSPACE_ROOT.name
    if ws_name and ws_name != "sandbox" and cleaned.startswith(f"{ws_name}/"):
        cleaned = cleaned.removeprefix(f"{ws_name}/").strip("/")
    return cleaned

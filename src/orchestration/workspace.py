from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Literal


REPO_ROOT = Path("/mnt/ai/langgraph-orchestrator").resolve()
ORCHESTRATOR_SRC = (REPO_ROOT / "src").resolve()
DEFAULT_WORKSPACE_ROOT = (REPO_ROOT / "sandbox").resolve()

AccessMode = Literal["sandbox_only", "read_external", "allow_external"]

_ACCESS_MODE_ALIASES: dict[str, AccessMode] = {
    "1": "sandbox_only",
    "sandbox_only": "sandbox_only",
    "sandbox": "sandbox_only",
    "2": "read_external",
    "read_external": "read_external",
    "default": "read_external",
    "3": "allow_external",
    "allow_external": "allow_external",
    "external": "allow_external",
}

_ACTIVE_WORKSPACE_ROOT: Path = DEFAULT_WORKSPACE_ROOT
_ACTIVE_ACCESS_MODE: AccessMode = "read_external"
_APPROVE_ALL_EXTERNAL: bool = False
_APPROVED_EXTERNAL_PATHS: set[Path] = set()
_EXTERNAL_READ_FILES: list[Path] = []
_PROMPT_CALLBACK: Callable[[str], str] = input


def configure_access_policy(
    mode: str | int = "read_external",
    *,
    approve_all_external: bool = False,
    external_read_files: list[str | Path] | None = None,
    prompt_callback: Callable[[str], str] | None = None,
) -> AccessMode:
    """
    Configure the 3-tier file access policy:
    - Mode 1 ('sandbox_only' / '1'): Read & write restricted strictly to `sandbox/`.
    - Mode 2 ('read_external' / '2', DEFAULT): Can read outside `sandbox/`, but can ONLY create/modify inside `sandbox/`.
    - Mode 3 ('allow_external' / '3'): Can read/write outside `sandbox/` with explicit user permission (`[y]es / [n]o / [a]ll`).
    """
    global _ACTIVE_ACCESS_MODE, _APPROVE_ALL_EXTERNAL, _APPROVED_EXTERNAL_PATHS, _EXTERNAL_READ_FILES, _PROMPT_CALLBACK

    key = str(mode).strip().lower()
    if key not in _ACCESS_MODE_ALIASES:
        raise ValueError(
            f"Invalid access mode '{mode}'. Use 1 ('sandbox_only'), 2 ('read_external'), or 3 ('allow_external')."
        )

    resolved_mode = _ACCESS_MODE_ALIASES[key]
    _ACTIVE_ACCESS_MODE = resolved_mode
    _APPROVE_ALL_EXTERNAL = bool(approve_all_external)
    _APPROVED_EXTERNAL_PATHS = set()
    _PROMPT_CALLBACK = prompt_callback or input

    # If switching to Mode 1 or Mode 2 while an external workspace was active, reset workspace to sandbox/
    if resolved_mode in ("sandbox_only", "read_external") and not is_inside_sandbox(_ACTIVE_WORKSPACE_ROOT):
        set_active_workspace(DEFAULT_WORKSPACE_ROOT)

    resolved_reads: list[Path] = []
    for raw_rf in external_read_files or []:
        p = Path(raw_rf).expanduser().resolve()
        check_read_permission(p)
        if p.is_file():
            resolved_reads.append(p)
        elif p.is_dir():
            for child in sorted(p.rglob("*")):
                if child.is_file() and not any(part.startswith(".") for part in child.relative_to(p).parts):
                    resolved_reads.append(child)
    _EXTERNAL_READ_FILES = resolved_reads

    return _ACTIVE_ACCESS_MODE


def get_access_mode() -> AccessMode:
    return _ACTIVE_ACCESS_MODE


def get_external_read_files() -> list[Path]:
    return list(_EXTERNAL_READ_FILES)


def is_inside_sandbox(path: Path) -> bool:
    """Return True if `path` is inside DEFAULT_WORKSPACE_ROOT (`sandbox/`)."""
    resolved = path.resolve()
    return resolved == DEFAULT_WORKSPACE_ROOT or resolved.is_relative_to(DEFAULT_WORKSPACE_ROOT)


def _check_not_Protected_orchestrator_path(resolved: Path) -> None:
    """Permanently block writing to '/', REPO_ROOT, or inside 'src/'."""
    if resolved == Path("/") or resolved == REPO_ROOT or resolved.is_relative_to(ORCHESTRATOR_SRC):
        raise ValueError(
            f"Refusing unsafe path '{resolved}': cannot modify '/', "
            f"the orchestrator root '{REPO_ROOT}', or '{ORCHESTRATOR_SRC}'."
        )


def check_read_permission(path: str | Path) -> Path:
    """
    Validate read access to `path` under the active AccessMode:
    - Mode 1 ('sandbox_only'): Must be inside `sandbox/`.
    - Mode 2 ('read_external') & Mode 3 ('allow_external'): Allowed inside and outside `sandbox/`.
    """
    candidate = Path(path).expanduser()
    resolved = candidate.resolve() if candidate.is_absolute() else (get_workspace_root() / candidate).resolve()

    if _ACTIVE_ACCESS_MODE == "sandbox_only" and not is_inside_sandbox(resolved):
        raise PermissionError(
            f"[ACCESS MODE 1: sandbox_only] Reading external file '{resolved}' outside "
            f"'{DEFAULT_WORKSPACE_ROOT}' is forbidden. Use Mode 2 ('read_external') or Mode 3 ('allow_external')."
        )
    return resolved


def _is_path_approved_for_external_write(resolved: Path) -> bool:
    if _APPROVE_ALL_EXTERNAL:
        return True
    for approved in _APPROVED_EXTERNAL_PATHS:
        if resolved == approved or resolved.is_relative_to(approved):
            return True
    return False


def request_external_write_permission(target: Path, operation: str = "create/modify") -> None:
    """
    Enforce write access control for `target`:
    - If `target` is inside `sandbox/`, always allowed.
    - If Mode 1 or Mode 2 is active, raise `PermissionError`.
    - If Mode 3 ('allow_external') is active, check prior approval or prompt the user
      with `[y]es / [n]o / [a]ll (yes to all)`.
    """
    global _APPROVE_ALL_EXTERNAL

    resolved = target.resolve()
    _check_not_Protected_orchestrator_path(resolved)

    if is_inside_sandbox(resolved):
        return

    if _ACTIVE_ACCESS_MODE in ("sandbox_only", "read_external"):
        raise PermissionError(
            f"[ACCESS MODE: {_ACTIVE_ACCESS_MODE}] Modifying or creating files outside sandbox "
            f"('{resolved}') is forbidden. Only files inside '{DEFAULT_WORKSPACE_ROOT}' may be modified. "
            f"To permit external writes with confirmation, use Mode 3 (--access-mode 3)."
        )

    if _is_path_approved_for_external_write(resolved):
        return

    prompt_msg = (
        f"\n[PERMISSION REQUIRED - Mode 3] Allow orchestrator to {operation} external path "
        f"'{resolved}' outside sandbox/? [y]es / [n]o / [a]ll (yes to all): "
    )
    try:
        answer = _PROMPT_CALLBACK(prompt_msg).strip().lower()
    except EOFError as exc:
        raise PermissionError(
            f"Non-interactive session denied external write to '{resolved}'. "
            f"Pass --yes-external (or approve_all_external=True) to pre-approve."
        ) from exc

    if answer in ("a", "all", "yes to all"):
        _APPROVE_ALL_EXTERNAL = True
        _APPROVED_EXTERNAL_PATHS.add(resolved)
        print(f"[PERMISSION] Approved all external writes for this session.")
        return

    if answer in ("y", "yes"):
        _APPROVED_EXTERNAL_PATHS.add(resolved)
        print(f"[PERMISSION] Approved external path: {resolved}")
        return

    raise PermissionError(f"User denied permission to {operation} external path: '{resolved}'.")


def set_active_workspace(
    path: str | Path | None,
    *,
    access_mode: str | int | None = None,
    approve_all_external: bool | None = None,
) -> Path:
    """
    Set and initialize the active target workspace directory.
    If `path` is outside `sandbox/`, enforces the active AccessMode (blocking in
    Modes 1 & 2, and prompting for `[y]es / [n]o / [a]ll` in Mode 3).
    """
    global _ACTIVE_WORKSPACE_ROOT, _ACTIVE_ACCESS_MODE, _APPROVE_ALL_EXTERNAL

    if access_mode is not None:
        key = str(access_mode).strip().lower()
        if key not in _ACCESS_MODE_ALIASES:
            raise ValueError(f"Invalid access mode: {access_mode}")
        _ACTIVE_ACCESS_MODE = _ACCESS_MODE_ALIASES[key]

    if approve_all_external is not None:
        _APPROVE_ALL_EXTERNAL = bool(approve_all_external)

    if path is None or str(path).strip() == "":
        resolved = DEFAULT_WORKSPACE_ROOT
    else:
        candidate = Path(path).expanduser()
        if not candidate.is_absolute():
            resolved = (Path.cwd() / candidate).resolve()
        else:
            resolved = candidate.resolve()

    _check_not_Protected_orchestrator_path(resolved)

    if not is_inside_sandbox(resolved):
        request_external_write_permission(resolved, operation="use external workspace directory")

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

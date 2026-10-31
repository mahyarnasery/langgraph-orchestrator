from pathlib import Path
import shutil

from src.orchestration.graph import _build_repository_map
from src.orchestration.tools import create_file, read_file, write_file
from src.orchestration.workspace import (
    DEFAULT_WORKSPACE_ROOT,
    configure_access_policy,
    get_access_mode,
    set_active_workspace,
)


EXTERNAL_DIR = Path("/tmp/langgraph_alpha11_external").resolve()
EXTERNAL_REF_FILE = EXTERNAL_DIR / "reference_spec.json"


def setup_external_fixture() -> None:
    if EXTERNAL_DIR.exists():
        shutil.rmtree(EXTERNAL_DIR, ignore_errors=True)
    EXTERNAL_DIR.mkdir(parents=True, exist_ok=True)
    EXTERNAL_REF_FILE.write_text('{"protocol": "LOHT-v2", "fps": 30}\n', encoding="utf-8")


def test_mode1_sandbox_only() -> None:
    """
    Mode 1 (`sandbox_only` / `1`):
    - Can read, create, and modify files inside `sandbox/`.
    - Cannot read files outside `sandbox/` (`PermissionError`).
    - Cannot set workspace or write files outside `sandbox/` (`PermissionError`).
    """
    print("\n================================================================")
    print("=== MODE 1: SANDBOX_ONLY (STRICT ISOLATION) ===")
    print("================================================================")
    setup_external_fixture()
    configure_access_policy("1")
    set_active_workspace(DEFAULT_WORKSPACE_ROOT)
    assert get_access_mode() == "sandbox_only"

    # 1. Reading & writing inside sandbox/ succeeds
    write_file.func(path="mode1_ok.txt", content="inside sandbox")
    res = read_file.func(path="mode1_ok.txt")
    assert res["content"].strip() == "inside sandbox"
    (DEFAULT_WORKSPACE_ROOT / "mode1_ok.txt").unlink(missing_ok=True)

    # 2. Reading external file outside sandbox/ is BLOCKED
    read_blocked = False
    try:
        read_file.func(path=str(EXTERNAL_REF_FILE))
    except PermissionError as exc:
        read_blocked = True
        print(f"Mode 1 correctly blocked external read: {exc}")
    assert read_blocked, "Mode 1 must block reading files outside sandbox/!"

    # 3. Setting workspace or writing outside sandbox/ is BLOCKED
    write_blocked = False
    try:
        set_active_workspace(EXTERNAL_DIR)
    except PermissionError as exc:
        write_blocked = True
        print(f"Mode 1 correctly blocked external workspace write: {exc}")
    assert write_blocked, "Mode 1 must block external workspace!"
    print("=== MODE 1 PASSED ===")


def test_mode2_read_external_default() -> None:
    """
    Mode 2 (`read_external` / `2` — DEFAULT):
    - Can read files outside `sandbox/` (both via `read_file` and `--read-external` repo map).
    - Can create and modify files inside `sandbox/`.
    - Cannot create or modify files outside `sandbox/` (`PermissionError`).
    """
    print("\n================================================================")
    print("=== MODE 2 (DEFAULT): READ_EXTERNAL, WRITE SANDBOX ONLY ===")
    print("================================================================")
    configure_access_policy(
        "2",
        external_read_files=[EXTERNAL_REF_FILE],
    )
    set_active_workspace(DEFAULT_WORKSPACE_ROOT)
    assert get_access_mode() == "read_external"

    # 1. Reading external file outside sandbox/ SUCCEEDS in Mode 2
    ext_res = read_file.func(path=str(EXTERNAL_REF_FILE))
    print(f"Mode 2 read external file successfully: {ext_res['content'].strip()}")
    assert '"protocol": "LOHT-v2"' in ext_res["content"]

    # 2. External read-only file appears in the repository map for Planner/Foreman
    repo_map = _build_repository_map()
    print(f"Mode 2 Repo Map with external read-only file:\n{repo_map}")
    assert "[EXTERNAL READ-ONLY]" in repo_map and "reference_spec.json" in repo_map

    # 3. Writing inside sandbox/ SUCCEEDS
    write_file.func(path="mode2_inside.txt", content="allowed inside sandbox")
    (DEFAULT_WORKSPACE_ROOT / "mode2_inside.txt").unlink(missing_ok=True)

    # 4. Attempting to set workspace or write outside sandbox/ is BLOCKED in Mode 2
    ext_write_blocked = False
    try:
        set_active_workspace(EXTERNAL_DIR)
    except PermissionError as exc:
        ext_write_blocked = True
        print(f"Mode 2 correctly blocked external write: {exc}")
    assert ext_write_blocked, "Mode 2 must block writing outside sandbox/!"
    print("=== MODE 2 PASSED ===")


def test_mode3_allow_external_with_interactive_prompt() -> None:
    """
    Mode 3 (`allow_external` / `3`):
    - Prompts the user before creating/modifying any path outside `sandbox/`:
      * `'n'` -> raises `PermissionError`
      * `'y'` -> approves that specific external path
      * `'a'` -> approves "yes to all" for the session without prompting again
    """
    print("\n================================================================")
    print("=== MODE 3: ALLOW_EXTERNAL WITH [y]es / [n]o / [a]ll PROMPT ===")
    print("================================================================")

    responses = iter(["n", "y", "a"])
    prompts_seen: list[str] = []

    def _mock_prompt(msg: str) -> str:
        prompts_seen.append(msg)
        ans = next(responses)
        print(f"{msg.strip()} -> {ans}")
        return ans

    configure_access_policy(
        "3",
        approve_all_external=False,
        prompt_callback=_mock_prompt,
    )
    assert get_access_mode() == "allow_external"

    # 1. First attempt: user answers 'n' -> must raise PermissionError
    denied = False
    try:
        set_active_workspace(EXTERNAL_DIR)
    except PermissionError as exc:
        denied = True
        print(f"Mode 3 ('n') correctly denied external access: {exc}")
    assert denied, "Expected PermissionError when user answers 'n'"

    # 2. Second attempt: user answers 'y' -> approves EXTERNAL_DIR
    ws = set_active_workspace(EXTERNAL_DIR)
    assert ws == EXTERNAL_DIR
    create_file.func(path="approved_file.txt", content="created in external workspace")
    assert (EXTERNAL_DIR / "approved_file.txt").is_file()

    # 3. Third attempt on a second external directory: user answers 'a' ("yes to all")
    second_ext = Path("/tmp/langgraph_alpha11_external_2").resolve()
    third_ext = Path("/tmp/langgraph_alpha11_external_3").resolve()
    try:
        set_active_workspace(second_ext)
        # Because 'a' ("yes to all") was chosen, switching to third_ext should NOT prompt again!
        prompts_before = len(prompts_seen)
        set_active_workspace(third_ext)
        assert len(prompts_seen) == prompts_before, "Expected 'a' (yes to all) to skip future prompts!"
        print("Mode 3 ('a' / yes to all) verified: subsequent external path required 0 extra prompts.")
    finally:
        shutil.rmtree(second_ext, ignore_errors=True)
        shutil.rmtree(third_ext, ignore_errors=True)

    print("=== MODE 3 PASSED ===")


def main() -> None:
    try:
        test_mode1_sandbox_only()
        test_mode2_read_external_default()
        test_mode3_allow_external_with_interactive_prompt()
        print("\n=== ALPHA11 ALL 3 ACCESS CONTROL MODES PASSED ===")
    finally:
        configure_access_policy("2")
        set_active_workspace(DEFAULT_WORKSPACE_ROOT)
        if EXTERNAL_DIR.exists():
            shutil.rmtree(EXTERNAL_DIR, ignore_errors=True)


if __name__ == "__main__":
    main()

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path
from typing import Iterable

from .schemas import TaskBatch, TaskResult, ValidationResult


REPO_ROOT = Path("/mnt/ai/langgraph-orchestrator").resolve()
WORKSPACE_ROOT = (REPO_ROOT / "sandbox").resolve()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _absolute(path: str) -> Path:
    """
    Convert a workspace-relative path into an absolute path.

    The resulting path must remain inside WORKSPACE_ROOT.
    """

    candidate = Path(path)

    if candidate.is_absolute():
        raise ValueError(
            f"Absolute paths are not allowed: {path}"
        )

    # Normalize sandbox prefix if present
    normalized = candidate
    if str(candidate).startswith("sandbox/"):
        normalized = Path(str(candidate).removeprefix("sandbox/"))

    resolved = (WORKSPACE_ROOT / normalized).resolve()

    try:
        resolved.relative_to(WORKSPACE_ROOT)
    except ValueError as exc:
        raise ValueError(
            f"Path escapes validation workspace: {path}"
        ) from exc

    return resolved


def _check_file_exists(
    path: str,
    checks: list[str],
    errors: list[str],
) -> bool:
    """Validate that a required file exists."""

    absolute = _absolute(path)

    if absolute.is_file():
        checks.append(
            f"exists: {path}"
        )
        return True

    if absolute.exists():
        errors.append(
            f"path is not a file: {path}"
        )
        return False

    errors.append(
        f"missing file: {path}"
    )
    return False


def _check_python_syntax(
    path: str,
    checks: list[str],
    errors: list[str],
) -> None:
    """
    Parse a Python file with ast.

    This is deterministic and does not execute the file.
    """

    absolute = _absolute(path)

    try:
        source = absolute.read_text(
            encoding="utf-8"
        )

        if not source.strip():
            errors.append(
                f"empty file: {path} has no content"
            )
            return

        ast.parse(source)

        checks.append(
            f"syntax OK: {path}"
        )

    except SyntaxError as exc:
        line_number = exc.lineno
        source_line = ""

        if line_number is not None:
            try:
                source_line = (
                    source.splitlines()[line_number - 1].strip()
                )
            except IndexError:
                source_line = ""

        if source_line:
            errors.append(
                f"syntax error in {path} "
                f"(line {line_number}): "
                f"{exc.msg}\n"
                f"Source: {source_line}"
            )
        else:
            errors.append(
                f"syntax error in {path} "
                f"(line {line_number}): "
                f"{exc.msg}"
            )

    except Exception as exc:
        errors.append(
            f"unable to parse {path}: {exc}"
        )


def _check_python_execution(
    path: str,
    checks: list[str],
    errors: list[str],
) -> None:
    """
    Execute a Python file deterministically using the same
    Python interpreter running the orchestrator.

    The script must:
    - exit successfully
    - produce no execution error
    """

    absolute = _absolute(path)

    try:
        completed = subprocess.run(
            [
                sys.executable,
                str(absolute),
            ],
            cwd=WORKSPACE_ROOT,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )

    except subprocess.TimeoutExpired:
        errors.append(
            f"execution timed out: {path}"
        )
        return

    except Exception as exc:
        errors.append(
            f"unable to execute {path}: {exc}"
        )
        return

    if completed.returncode != 0:
        stderr = completed.stderr.strip()

        if stderr:
            errors.append(
                f"execution failed: {path} "
                f"(exit code {completed.returncode})\n"
                f"{stderr}"
            )
        else:
            errors.append(
                f"execution failed: {path} "
                f"(exit code {completed.returncode})"
            )

        return

    output = completed.stdout.strip()

    checks.append(
        f"execution OK: {path}"
    )

    if output:
        checks.append(
            f"stdout: {output}"
        )


def _check_scope(
    batch: TaskBatch,
    task_results: Iterable[TaskResult],
    checks: list[str],
    errors: list[str],
) -> None:
    """
    Ensure workers only modified files declared in the batch.
    """

    allowed = set(batch.relevant_files)

    for result in task_results:
        for changed in result.files_changed:
            if changed in allowed:
                checks.append(
                    f"scope OK: {changed}"
                )
            else:
                errors.append(
                    f"scope violation: {changed} "
                    f"modified outside batch scope"
                )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def validate_batch(
    batch: TaskBatch,
    task_results: list[TaskResult],
) -> ValidationResult:
    """
    Deterministically validate a completed TaskBatch.

    Validators:
    - required file existence
    - Python syntax
    - Python execution
    - worker scope compliance
    """

    checks: list[str] = []
    errors: list[str] = []

    # --------------------------------------------------------
    # Validate repository workspace files
    # --------------------------------------------------------

    for path in batch.relevant_files:
        exists = _check_file_exists(
            path,
            checks,
            errors,
        )

        if not exists:
            continue

        if path.endswith(".py"):
            _check_python_syntax(
                path,
                checks,
                errors,
            )

            # Only execute if syntax is valid and file not empty.
            syntax_failed = any(
                error.startswith(f"syntax error in {path}")
                or error.startswith(f"empty file: {path}")
                for error in errors
            )

            if not syntax_failed:
                _check_python_execution(
                    path,
                    checks,
                    errors,
                )

    # --------------------------------------------------------
    # Validate Worker scope
    # --------------------------------------------------------

    _check_scope(
        batch,
        task_results,
        checks,
        errors,
    )

    changed_files = sorted(
        {
            file
            for result in task_results
            for file in result.files_changed
        }
    )

    return ValidationResult(
        passed=len(errors) == 0,
        checks=checks,
        errors=errors,
        files_checked=batch.relevant_files,
        changed_files=changed_files,
    )

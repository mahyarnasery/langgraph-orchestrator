from __future__ import annotations

import ast
from pathlib import Path
from typing import Iterable

from .schemas import TaskBatch, TaskResult, ValidationResult

# Repository sandbox root
SANDBOX_ROOT = Path("/mnt/ai/langgraph-orchestrator/sandbox")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _absolute(path: str) -> Path:
    """Convert a sandbox-relative path into an absolute filesystem path."""
    return SANDBOX_ROOT / path


def _check_file_exists(path: str, checks: list[str], errors: list[str]) -> bool:
    """Validate that a required file exists."""
    absolute = _absolute(path)

    if absolute.exists():
        checks.append(f"exists: {path}")
        return True

    errors.append(f"missing file: {path}")
    return False


def _check_python_syntax(path: str, checks: list[str], errors: list[str]) -> None:
    """
    Parse a Python file with ast.

    This is deterministic and catches syntax errors without executing code.
    """
    absolute = _absolute(path)

    try:
        source = absolute.read_text(encoding="utf-8")
        ast.parse(source)
        checks.append(f"syntax OK: {path}")

    except SyntaxError as exc:
        errors.append(
            f"syntax error in {path} (line {exc.lineno}): {exc.msg}"
        )

    except Exception as exc:
        errors.append(f"unable to parse {path}: {exc}")


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
                checks.append(f"scope OK: {changed}")
            else:
                errors.append(
                    f"scope violation: {changed} modified outside batch scope"
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

    Current validators:
    - required file existence
    - Python syntax (for *.py)
    - worker scope compliance

    No LLM reasoning occurs here.
    """
    checks: list[str] = []
    errors: list[str] = []

    # Validate repository files
    for path in batch.relevant_files:
        exists = _check_file_exists(path, checks, errors)

        if exists and path.endswith(".py"):
            _check_python_syntax(path, checks, errors)

    # Validate worker scope
    _check_scope(batch, task_results, checks, errors)

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
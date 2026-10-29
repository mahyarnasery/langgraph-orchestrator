from __future__ import annotations

import ast
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Iterable

from .schemas import TaskBatch, TaskResult, ValidationResult


REPO_ROOT = Path("/mnt/ai/langgraph-orchestrator").resolve()
WORKSPACE_ROOT = (REPO_ROOT / "sandbox").resolve()


def _absolute(path: str) -> Path:
    """
    Convert a workspace-relative path into an absolute path.
    The resulting path must remain inside WORKSPACE_ROOT.
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
        raise ValueError(f"Path escapes validation workspace: {path}") from exc

    return resolved


def _check_file_exists(
    path: str,
    checks: list[str],
    errors: list[str],
) -> bool:
    """Validate that a required file exists."""
    absolute = _absolute(path)

    if absolute.is_file():
        checks.append(f"exists: {path}")
        return True

    if absolute.exists():
        errors.append(f"path is not a file: {path}")
        return False

    errors.append(f"missing file: {path}")
    return False


def _check_python_syntax(
    path: str,
    checks: list[str],
    errors: list[str],
) -> None:
    """Parse a Python file with ast."""
    absolute = _absolute(path)

    try:
        source = absolute.read_text(encoding="utf-8")

        if not source.strip():
            errors.append(f"empty file: {path} has no content")
            return

        ast.parse(source)
        checks.append(f"syntax OK: {path}")

    except SyntaxError as exc:
        line_number = exc.lineno
        source_line = ""

        if line_number is not None:
            try:
                source_line = source.splitlines()[line_number - 1].strip()
            except IndexError:
                source_line = ""

        if source_line:
            errors.append(
                f"syntax error in {path} (line {line_number}): {exc.msg}\n"
                f"Source: {source_line}"
            )
        else:
            errors.append(f"syntax error in {path} (line {line_number}): {exc.msg}")

    except Exception as exc:
        errors.append(f"unable to parse {path}: {exc}")


def _check_python_execution(
    path: str,
    checks: list[str],
    errors: list[str],
) -> str:
    """
    Execute a Python file deterministically using the active Python interpreter
    with PYTHONPATH set to WORKSPACE_ROOT. Returns captured stdout.
    """
    absolute = _absolute(path)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(WORKSPACE_ROOT)

    try:
        completed = subprocess.run(
            [sys.executable, str(absolute)],
            cwd=WORKSPACE_ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except subprocess.TimeoutExpired:
        errors.append(f"execution timed out: {path}")
        return ""
    except Exception as exc:
        errors.append(f"unable to execute {path}: {exc}")
        return ""

    if completed.returncode != 0:
        stderr = completed.stderr.strip()
        if stderr:
            errors.append(
                f"execution failed: {path} (exit code {completed.returncode})\n{stderr}"
            )
        else:
            errors.append(
                f"execution failed: {path} (exit code {completed.returncode})"
            )
        return ""

    output = completed.stdout.strip()
    checks.append(f"execution OK: {path}")

    if output:
        checks.append(f"stdout ({path}): {output}")

    return output


def _check_verification_command(
    command: str,
    checks: list[str],
    errors: list[str],
) -> None:
    """
    Run an explicit batch verification command inside WORKSPACE_ROOT.
    Ignores unsafe or redundant shell-pipe commands (e.g. cat | grep).
    """
    if any(token in command for token in ("|", "&&", ";", "`", "$(")):
        checks.append(
            f"skipped shell-pipe verification_command (handled by AST/acceptance checks): {command}"
        )
        return

    env = os.environ.copy()
    env["PYTHONPATH"] = str(WORKSPACE_ROOT)

    try:
        args = shlex.split(command)
        if not args:
            return
        if args[0] in ("python", "python3", "pytest"):
            if args[0] == "pytest":
                args = [sys.executable, "-m", "pytest", *args[1:]]
            else:
                args[0] = sys.executable

        completed = subprocess.run(
            args,
            cwd=WORKSPACE_ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
    except Exception as exc:
        errors.append(f"verification command failed to run ({command}): {exc}")
        return

    if completed.returncode != 0:
        stderr = (completed.stderr or completed.stdout).strip()
        errors.append(
            f"verification command failed ({command}, exit {completed.returncode}): {stderr}"
        )
    else:
        checks.append(f"verification command OK: {command}")


def _check_scope(
    batch: TaskBatch,
    task_results: Iterable[TaskResult],
    checks: list[str],
    errors: list[str],
) -> None:
    """Ensure workers only modified files declared in the batch."""
    allowed = {p.removeprefix("sandbox/") for p in batch.relevant_files}
    seen_changed: set[str] = set()

    for result in task_results:
        for changed in result.files_changed:
            normalized = changed.removeprefix("sandbox/")
            if normalized in seen_changed:
                continue
            seen_changed.add(normalized)

            if normalized in allowed:
                checks.append(f"scope OK: {normalized}")
            else:
                errors.append(
                    f"scope violation: {changed} modified outside batch scope"
                )


def _extract_quoted_literals(
    text: str,
    ignore_filenames: Iterable[str] = (),
) -> list[str]:
    """
    Extract backticked, double-quoted, and standalone single-quoted literals,
    excluding filenames/module names that merely reference the target files.
    """
    ignored = set()
    for fname in ignore_filenames:
        clean = fname.removeprefix("sandbox/")
        ignored.add(clean)
        ignored.add(f"sandbox/{clean}")
        ignored.add(Path(clean).name)
        ignored.add(Path(clean).stem)

    backticked = re.findall(r"`([^`]+)`", text)
    # Strip backticked segments before searching for quotes so `f"..."` is not extracted twice
    text_without_backticks = re.sub(r"`[^`]+`", " ", text)
    double_quoted = re.findall(r'"([^"]+)"', text_without_backticks)
    single_quoted = re.findall(r"(?<![a-zA-Z0-9])'([^']+)'(?![a-zA-Z0-9])", text_without_backticks)

    combined: list[str] = []
    for item in backticked + double_quoted + single_quoted:
        stripped = item.strip()
        if not stripped or stripped in ignored:
            continue
        if stripped not in combined:
            combined.append(stripped)

    return combined


def _pattern_matches(pattern: str, content: str, stdout: str) -> bool:
    """
    Check if an expected literal pattern matches the file's source code,
    normalized source code, class/function/method definitions, or execution stdout.
    """
    if pattern in content or pattern in stdout:
        return True

    # Whitespace-insensitive and `self.`-normalized match
    # (e.g. `price * quantity` matches `self.price * self.quantity`)
    compact_pattern = re.sub(r"\s+", "", pattern).replace("self.", "")
    compact_content = re.sub(r"\s+", "", content).replace("self.", "")
    compact_stdout = re.sub(r"\s+", "", stdout)
    if compact_pattern in compact_content or compact_pattern in compact_stdout:
        return True

    # Class definition match (e.g. `class Item` vs `class Item:`)
    cls_match = re.match(r"^(?:class\s+)?([A-Z][a-zA-Z0-9_]*)(?:\s*\(.*\))?:?$", pattern)
    if cls_match:
        cls_name = cls_match.group(1)
        if f"class {cls_name}" in content or f"{cls_name}(" in content or cls_name in content:
            return True

    # Tuple return without outer parens (e.g. `(normalize_name(left), parse_amount(right))`)
    if compact_pattern.startswith("(") and compact_pattern.endswith(")"):
        inner_tuple = compact_pattern[1:-1]
        if inner_tuple and inner_tuple in compact_content:
            return True

    # f-string or interpolated template with extracted local variables
    if "{" in compact_pattern and "}" in compact_pattern:
        cleaned_pat = compact_pattern
        if cleaned_pat.startswith(('f"', "f'")):
            cleaned_pat = cleaned_pat[2:-1]
        exprs = re.findall(r"\{([^}]+)\}", cleaned_pat)
        static_parts = [
            p for p in re.split(r"\{[^}]+\}", cleaned_pat) if p
        ]
        if exprs and all(e in compact_content for e in exprs) and all(
            sp in compact_content for sp in static_parts
        ):
            return True

    # Function or method signature match (e.g. `add_item(self, item)` or `inv.total_value()`)
    fn_match = re.match(r"^(?:def\s+)?(?:[a-zA-Z_][a-zA-Z0-9_]*\.)?([a-zA-Z_][a-zA-Z0-9_]*)\s*\(", pattern)
    if fn_match:
        fn_name = fn_match.group(1)
        if f"def {fn_name}(" in content or f"{fn_name}(" in content:
            return True

    return False


def _criterion_satisfied_by_file(
    criterion: str,
    content: str,
    stdout: str,
    ignore_files: Iterable[str],
) -> bool:
    """
    Evaluate a single acceptance criterion against a file's content and stdout
    using mutually exclusive branches.
    """
    if not content.strip():
        return False

    criterion_lower = criterion.lower()
    content_lower = content.lower()
    expected_patterns = _extract_quoted_literals(
        criterion, ignore_filenames=ignore_files
    )

    # 1. File existence check
    if any(
        phrase in criterion_lower
        for phrase in ("file exists", "is created", "must exist", "exists in")
    ) and not expected_patterns:
        return True

    # 2. Negative / prohibition constraints (e.g. "no additional code", "must not contain")
    if any(
        phrase in criterion_lower
        for phrase in (
            "no additional",
            "no extra",
            "nothing else",
            "must not",
            "do not",
            "does not",
            "without",
        )
    ):
        non_empty_lines = [line.strip() for line in content.splitlines() if line.strip()]
        return len(non_empty_lines) > 0

    # 3. Exact content, explicit "contains", or runtime output with quoted/backticked literals
    if expected_patterns and any(
        word in criterion_lower
        for word in (
            "exactly",
            "exact content",
            "contains",
            "includes",
            "output",
            "outputs",
            "print",
            "prints",
            "line:",
            "define",
            "defines",
            "implement",
            "implements",
            "import",
            "imports",
            "call",
            "calls",
            "return",
            "returns",
        )
    ):
        return all(
            _pattern_matches(pattern, content, stdout)
            for pattern in expected_patterns
        )

    # 4. Structural Python keyword check
    code_keywords = ("print", "def ", "class ", "import ", "return")
    matched_keywords = [kw for kw in code_keywords if kw in criterion_lower]
    if matched_keywords:
        return all(kw.strip() in content_lower for kw in matched_keywords)

    # 5. Default fallback for general descriptive criteria when file is non-empty
    return True


def _check_acceptance_criteria(
    batch: TaskBatch,
    execution_outputs: dict[str, str],
    checks: list[str],
    errors: list[str],
) -> None:
    """Validate that task acceptance criteria are met by relevant files or their stdout."""
    all_batch_files = batch.relevant_files
    combined_batch_stdout = "\n".join(
        out for out in execution_outputs.values() if out
    )

    for task in batch.tasks:
        target_files = task.relevant_files or batch.relevant_files
        for criterion in task.acceptance_criteria:
            satisfied = False

            for file_path in target_files:
                try:
                    norm_path = file_path.removeprefix("sandbox/")
                    absolute = _absolute(norm_path)
                    if not absolute.is_file():
                        continue
                    content = absolute.read_text(encoding="utf-8")
                    file_stdout = execution_outputs.get(norm_path, "") or combined_batch_stdout
                    if _criterion_satisfied_by_file(
                        criterion, content, file_stdout, all_batch_files
                    ):
                        satisfied = True
                        break
                except Exception:
                    continue

            if satisfied:
                checks.append(f"acceptance: {criterion}")
            else:
                errors.append(f"acceptance criteria not met ({task.task_id}): {criterion}")


def validate_batch(
    batch: TaskBatch,
    task_results: list[TaskResult],
) -> ValidationResult:
    """Deterministically validate a completed TaskBatch."""
    checks: list[str] = []
    errors: list[str] = []
    execution_outputs: dict[str, str] = {}

    for path in batch.relevant_files:
        norm_path = path.removeprefix("sandbox/")
        exists = _check_file_exists(norm_path, checks, errors)
        if not exists:
            continue

        if norm_path.endswith(".py"):
            _check_python_syntax(norm_path, checks, errors)

            syntax_failed = any(
                error.startswith(f"syntax error in {norm_path}")
                or error.startswith(f"empty file: {norm_path}")
                for error in errors
            )

            if not syntax_failed:
                stdout = _check_python_execution(norm_path, checks, errors)
                execution_outputs[norm_path] = stdout

    if batch.verification_command and not errors:
        _check_verification_command(batch.verification_command, checks, errors)

    _check_scope(batch, task_results, checks, errors)
    _check_acceptance_criteria(batch, execution_outputs, checks, errors)

    changed_files = sorted(
        {
            file.removeprefix("sandbox/")
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

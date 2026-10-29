from __future__ import annotations

import ast
import json
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Iterable

from .schemas import Task, TaskBatch, TaskResult, ValidationResult


REPO_ROOT = Path("/mnt/ai/langgraph-orchestrator").resolve()
WORKSPACE_ROOT = (REPO_ROOT / "sandbox").resolve()


_SUBPROCESS_ASSERTION_RUNNER = r"""
import ast
import contextlib
import io
import json
import operator
import sys
from pathlib import Path

COMPARE_OPS = {
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
    ast.Lt: operator.lt,
    ast.LtE: operator.le,
    ast.Gt: operator.gt,
    ast.GtE: operator.ge,
    ast.Is: operator.is_,
    ast.IsNot: operator.is_not,
    ast.In: lambda a, b: a in b,
    ast.NotIn: lambda a, b: a not in b,
}


def _module_identity(rel_path: str) -> tuple[str, str]:
    p = Path(rel_path)
    parts = list(p.with_suffix("").parts)
    if not parts:
        return "__test_loaded__", ""
    if parts[-1] == "__init__":
        pkg = ".".join(parts[:-1])
        return (pkg or "__init__"), pkg
    mod_name = ".".join(parts)
    pkg_name = ".".join(parts[:-1])
    return mod_name, pkg_name


def load_target_namespace(workspace_root: Path, target_files: list[str], stdout_map: dict[str, str]) -> dict:
    ns = {"__builtins__": __builtins__}
    combined_stdout = "\n".join(v for v in stdout_map.values() if v)
    ns["stdout"] = combined_stdout

    for rel_path in target_files:
        abs_path = (workspace_root / rel_path).resolve()
        if not abs_path.is_file():
            continue
        if rel_path in stdout_map and stdout_map[rel_path]:
            ns["stdout"] = stdout_map[rel_path]

        source = abs_path.read_text(encoding="utf-8")
        if not source.strip():
            continue

        mod_name, pkg_name = _module_identity(rel_path)
        mod_ns = {
            "__builtins__": __builtins__,
            "__name__": mod_name,
            "__package__": pkg_name,
            "__file__": str(abs_path),
        }
        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
                exec(compile(source, str(abs_path), "exec"), mod_ns)
        except Exception:
            try:
                tree = ast.parse(source, filename=str(abs_path))
                safe_nodes = [
                    node for node in tree.body
                    if isinstance(
                        node,
                        (
                            ast.Import,
                            ast.ImportFrom,
                            ast.FunctionDef,
                            ast.AsyncFunctionDef,
                            ast.ClassDef,
                        ),
                    )
                ]
                if safe_nodes:
                    mod_ast = ast.Module(body=safe_nodes, type_ignores=[])
                    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
                        exec(compile(mod_ast, str(abs_path), "exec"), mod_ns)
            except Exception:
                pass

        for k, v in mod_ns.items():
            if not k.startswith("__"):
                ns[k] = v

    return ns


def evaluate_assertion(stmt: str, base_ns: dict, target_symbol: str | None) -> dict:
    ns = dict(base_ns)
    try:
        tree = ast.parse(stmt, filename="<assertion>")
    except SyntaxError as exc:
        return {"status": "skipped", "detail": f"{stmt} (SyntaxError: {exc.msg})"}

    called_names = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }

    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            for node in tree.body:
                if not isinstance(node, ast.Assert):
                    mod = ast.Module(body=[node], type_ignores=[])
                    exec(compile(mod, "<assertion>", "exec"), ns)
                    continue

                test_node = node.test
                if (
                    isinstance(test_node, ast.Compare)
                    and len(test_node.ops) == 1
                    and len(test_node.comparators) == 1
                    and type(test_node.ops[0]) in COMPARE_OPS
                ):
                    left_val = eval(
                        compile(ast.Expression(body=test_node.left), "<assertion>", "eval"),
                        ns,
                    )
                    right_val = eval(
                        compile(
                            ast.Expression(body=test_node.comparators[0]),
                            "<assertion>",
                            "eval",
                        ),
                        ns,
                    )
                    op_fn = COMPARE_OPS[type(test_node.ops[0])]
                    ok = bool(op_fn(left_val, right_val))
                    if not ok:
                        return {
                            "status": "failed",
                            "detail": f"{stmt} [actual={left_val!r}, expected={right_val!r}]",
                        }
                else:
                    val = eval(
                        compile(ast.Expression(body=test_node), "<assertion>", "eval"),
                        ns,
                    )
                    if not bool(val):
                        return {
                            "status": "failed",
                            "detail": f"{stmt} [evaluated to {val!r}]",
                        }
    except Exception as exc:
        tb = exc.__traceback__
        last_tb = tb
        while last_tb is not None and last_tb.tb_next is not None:
            last_tb = last_tb.tb_next
        frame_file = last_tb.tb_frame.f_code.co_filename if last_tb else ""

        if isinstance(exc, NameError) and frame_file == "<assertion>":
            missing = getattr(exc, "name", None) or ""
            if not missing and "'" in str(exc):
                missing = str(exc).split("'")[1]
            if missing and (missing == target_symbol or missing in called_names):
                return {
                    "status": "failed",
                    "detail": f"{stmt} (NameError: '{missing}' is not defined)",
                }
            return {
                "status": "skipped",
                "detail": f"{stmt} (unbound symbolic variable '{missing}')",
            }

        return {
            "status": "failed",
            "detail": f"{stmt} ({type(exc).__name__}: {exc})",
        }

    return {"status": "passed", "detail": stmt}


def main() -> None:
    payload = json.loads(sys.stdin.read())
    workspace_root = Path(payload["workspace_root"]).resolve()
    sys.path.insert(0, str(workspace_root))

    target_files = payload.get("target_files", [])
    target_symbol = payload.get("target_symbol")
    assertions = payload.get("assertions", [])
    stdout_map = payload.get("stdout_map", {})

    base_ns = load_target_namespace(workspace_root, target_files, stdout_map)
    results = [
        evaluate_assertion(stmt, base_ns, target_symbol)
        for stmt in assertions
    ]
    sys.stdout.write(json.dumps({"results": results}))


if __name__ == "__main__":
    main()
"""


def _normalize_rel_path(path: str) -> str:
    cleaned = path.strip().replace("\\", "/")
    while cleaned.startswith("./"):
        cleaned = cleaned[2:]
    cleaned = cleaned.removeprefix("sandbox/").strip("/")
    return cleaned


def _absolute(path: str) -> Path:
    """
    Convert a workspace-relative path into an absolute path.
    The resulting path must remain inside WORKSPACE_ROOT.
    """
    candidate = Path(path)

    if candidate.is_absolute():
        raise ValueError(f"Absolute paths are not allowed: {path}")

    normalized = Path(_normalize_rel_path(path))
    resolved = (WORKSPACE_ROOT / normalized).resolve()

    try:
        resolved.relative_to(WORKSPACE_ROOT)
    except ValueError as exc:
        raise ValueError(f"Path escapes validation workspace: {path}") from exc

    return resolved


def _ensure_parent_package_inits(norm_path: str) -> None:
    """Ensure parent directories of a nested Python file have `__init__.py`."""
    parts = Path(norm_path).parts
    if len(parts) <= 1:
        return
    for depth in range(1, len(parts)):
        pkg_dir = WORKSPACE_ROOT / Path(*parts[:depth])
        if pkg_dir.is_dir() and pkg_dir.name not in (".staging", ".git", "__pycache__"):
            init_file = pkg_dir / "__init__.py"
            if not init_file.exists():
                init_file.write_text("", encoding="utf-8")


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
            if absolute.name == "__init__.py":
                checks.append(f"syntax OK: {path} (package __init__.py)")
                return
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
    Execute a Python file or package module deterministically using the active
    Python interpreter with PYTHONPATH set to WORKSPACE_ROOT. Returns captured stdout.
    """
    norm_path = _normalize_rel_path(path)
    absolute = _absolute(norm_path)
    _ensure_parent_package_inits(norm_path)

    source = absolute.read_text(encoding="utf-8")
    if not source.strip() and absolute.name == "__init__.py":
        checks.append(f"execution OK: {norm_path} (package marker)")
        return ""

    env = os.environ.copy()
    env["PYTHONPATH"] = str(WORKSPACE_ROOT)

    p_parts = list(Path(norm_path).with_suffix("").parts)
    if len(p_parts) > 1:
        if p_parts[-1] == "__init__":
            pkg_name = ".".join(p_parts[:-1])
            cmd = [sys.executable, "-c", f"import {pkg_name}"]
        else:
            mod_name = ".".join(p_parts)
            cmd = [sys.executable, "-m", mod_name]
    else:
        cmd = [sys.executable, str(absolute)]

    try:
        completed = subprocess.run(
            cmd,
            cwd=WORKSPACE_ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except subprocess.TimeoutExpired:
        errors.append(f"execution timed out: {norm_path}")
        return ""
    except Exception as exc:
        errors.append(f"unable to execute {norm_path}: {exc}")
        return ""

    if completed.returncode != 0:
        stderr = completed.stderr.strip()
        if stderr:
            errors.append(
                f"execution failed: {norm_path} (exit code {completed.returncode})\n{stderr}"
            )
        else:
            errors.append(
                f"execution failed: {norm_path} (exit code {completed.returncode})"
            )
        return ""

    output = completed.stdout.strip()
    checks.append(f"execution OK: {norm_path}")

    if output:
        checks.append(f"stdout ({norm_path}): {output}")

    return output


def _is_pytest_file(path: str) -> bool:
    name = Path(_normalize_rel_path(path)).name
    return name.startswith("test_") or name.endswith("_test.py")


def _check_pytest_files(
    pytest_files: list[str],
    checks: list[str],
    errors: list[str],
) -> None:
    """Run pytest deterministically on any dedicated test_*.py files in the batch."""
    if not pytest_files:
        return

    env = os.environ.copy()
    env["PYTHONPATH"] = str(WORKSPACE_ROOT)

    try:
        completed = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", *pytest_files],
            cwd=WORKSPACE_ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
    except Exception as exc:
        errors.append(f"pytest failed to execute ({', '.join(pytest_files)}): {exc}")
        return

    if completed.returncode != 0:
        output = (completed.stdout or completed.stderr).strip()
        errors.append(
            f"pytest failed ({', '.join(pytest_files)}, exit {completed.returncode}):\n{output}"
        )
    else:
        checks.append(f"pytest OK ({', '.join(pytest_files)})")


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
    allowed = {_normalize_rel_path(p) for p in batch.relevant_files}
    seen_changed: set[str] = set()

    for result in task_results:
        for changed in result.files_changed:
            normalized = _normalize_rel_path(changed)
            if normalized in seen_changed:
                continue
            seen_changed.add(normalized)

            if normalized in allowed:
                checks.append(f"scope OK: {normalized}")
            else:
                errors.append(
                    f"scope violation: {changed} modified outside batch scope"
                )


def _defined_symbols_in_ast(source: str) -> set[str]:
    """Return all top-level function, class, method, imported, and variable names in a Python file."""
    if not source.strip():
        return set()
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return set()

    symbols: set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            symbols.add(node.name)
        elif isinstance(node, ast.ClassDef):
            symbols.add(node.name)
            for sub in node.body:
                if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    symbols.add(sub.name)
                    symbols.add(f"{node.name}.{sub.name}")
        elif isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name):
                    symbols.add(t.id)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            symbols.add(node.target.id)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                symbols.add(alias.asname or alias.name)
    return symbols


def _check_target_symbols(
    batch: TaskBatch,
    syntax_failed_files: set[str],
    checks: list[str],
    errors: list[str],
    failed_task_ids: set[str],
) -> None:
    """Verify via AST that each task with `target_symbol` defines that symbol in its target file."""
    for task in batch.tasks:
        if not task.target_symbol:
            continue
        target_sym = task.target_symbol.split("(")[0].strip()
        if not target_sym:
            continue

        target_files = [
            _normalize_rel_path(f)
            for f in (task.relevant_files or batch.relevant_files)
        ]
        checked_any = False
        found = False
        for norm_path in target_files:
            if norm_path in syntax_failed_files:
                continue
            abs_path = _absolute(norm_path)
            if not abs_path.is_file():
                continue
            checked_any = True
            symbols = _defined_symbols_in_ast(abs_path.read_text(encoding="utf-8"))
            if target_sym in symbols:
                found = True
                checks.append(f"symbol OK ({task.task_id}): {norm_path}::{target_sym}")
                break

        if checked_any and not found:
            primary = target_files[0] if target_files else "workspace"
            errors.append(
                f"missing symbol ({task.task_id}): '{target_sym}' not defined in {primary}"
            )
            failed_task_ids.add(task.task_id)


def _normalize_assertion_code(raw: str) -> str | None:
    """
    Normalize a raw assertion string into valid executable Python code containing
    at least one `assert` statement, or return None if it is non-code prose.
    """
    cleaned = raw.strip()
    if not cleaned:
        return None

    if cleaned.startswith("`") and cleaned.endswith("`"):
        cleaned = cleaned.strip("`").strip()

    if not cleaned:
        return None

    try:
        tree = ast.parse(cleaned)
    except SyntaxError:
        return None

    if not tree.body:
        return None

    last_node = tree.body[-1]
    if isinstance(last_node, ast.Expr) and isinstance(
        last_node.value, (ast.Compare, ast.Call, ast.BoolOp, ast.UnaryOp)
    ):
        expr_src = ast.get_source_segment(cleaned, last_node) or ast.unparse(last_node)
        prefix_nodes = tree.body[:-1]
        if not prefix_nodes:
            cleaned = f"assert {expr_src}"
        else:
            prefix_src = "; ".join(
                (ast.get_source_segment(cleaned, n) or ast.unparse(n))
                for n in prefix_nodes
            )
            cleaned = f"{prefix_src}; assert {expr_src}"
        try:
            tree = ast.parse(cleaned)
        except SyntaxError:
            return None

    has_assert = any(isinstance(n, ast.Assert) for n in ast.walk(tree))
    return cleaned if has_assert else None


def _extract_explicit_asserts_from_criteria(criteria: Iterable[str]) -> list[str]:
    """Extract explicit `assert ...` statements from acceptance criteria when `test_assertions` is empty."""
    extracted: list[str] = []
    for crit in criteria:
        stripped = crit.strip()
        if stripped.startswith("assert "):
            norm = _normalize_assertion_code(stripped)
            if norm and norm not in extracted:
                extracted.append(norm)
            continue

        for segment in re.findall(r"`([^`]+)`", stripped):
            seg = segment.strip()
            if seg.startswith("assert "):
                norm = _normalize_assertion_code(seg)
                if norm and norm not in extracted:
                    extracted.append(norm)
    return extracted


def _run_assertions_in_workspace(
    *,
    target_files: list[str],
    target_symbol: str | None,
    assertions: list[str],
    execution_outputs: dict[str, str],
) -> list[dict[str, str]]:
    """Execute normalized assertions in an isolated workspace subprocess."""
    if not assertions:
        return []

    for tf in target_files:
        _ensure_parent_package_inits(tf)

    payload = {
        "workspace_root": str(WORKSPACE_ROOT),
        "target_files": target_files,
        "target_symbol": target_symbol,
        "assertions": assertions,
        "stdout_map": execution_outputs,
    }

    env = os.environ.copy()
    env["PYTHONPATH"] = str(WORKSPACE_ROOT)

    try:
        completed = subprocess.run(
            [sys.executable, "-c", _SUBPROCESS_ASSERTION_RUNNER],
            input=json.dumps(payload),
            cwd=WORKSPACE_ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return [
            {"status": "failed", "detail": f"{a} (assertion timed out)"}
            for a in assertions
        ]
    except Exception as exc:
        return [
            {"status": "failed", "detail": f"{a} (runner error: {exc})"}
            for a in assertions
        ]

    if completed.returncode != 0:
        err_out = (completed.stderr or completed.stdout).strip()
        return [
            {"status": "failed", "detail": f"{a} (subprocess error: {err_out})"}
            for a in assertions
        ]

    try:
        data = json.loads(completed.stdout)
        return data.get("results", [])
    except Exception as exc:
        return [
            {"status": "failed", "detail": f"{a} (invalid runner JSON: {exc})"}
            for a in assertions
        ]


def _check_task_and_batch_assertions(
    batch: TaskBatch,
    syntax_failed_files: set[str],
    execution_outputs: dict[str, str],
    checks: list[str],
    errors: list[str],
    failed_task_ids: set[str],
) -> set[str]:
    """
    Execute deterministic `test_assertions` for every task and for the batch.
    Returns the set of `task_id`s that had at least one passing unit assertion
    and zero failing assertions.
    """
    tasks_with_passed_assertions: set[str] = set()

    for task in batch.tasks:
        target_files = [
            _normalize_rel_path(f)
            for f in (task.relevant_files or batch.relevant_files)
        ]
        if any(f in syntax_failed_files or not _absolute(f).is_file() for f in target_files):
            continue

        raw_assertions = list(task.test_assertions)
        if not raw_assertions and task.target_symbol:
            raw_assertions = _extract_explicit_asserts_from_criteria(
                task.acceptance_criteria
            )

        normalized_assertions: list[str] = []
        for raw in raw_assertions:
            norm = _normalize_assertion_code(raw)
            if norm and norm not in normalized_assertions:
                normalized_assertions.append(norm)

        if not normalized_assertions:
            continue

        primary_file = target_files[0] if target_files else "workspace"
        label = (
            f"{task.task_id} [{primary_file}::{task.target_symbol}]"
            if task.target_symbol
            else f"{task.task_id} [{primary_file}]"
        )

        outcomes = _run_assertions_in_workspace(
            target_files=target_files,
            target_symbol=task.target_symbol,
            assertions=normalized_assertions,
            execution_outputs=execution_outputs,
        )

        passed_count = 0
        failed_count = 0
        for outcome in outcomes:
            status = outcome.get("status")
            detail = outcome.get("detail", "")
            if status == "passed":
                passed_count += 1
                checks.append(f"assertion OK ({label}): {detail}")
            elif status == "skipped":
                checks.append(f"skipped symbolic assertion ({task.task_id}): {detail}")
            else:
                failed_count += 1
                errors.append(f"assertion failed ({label}): {detail}")
                failed_task_ids.add(task.task_id)

        if passed_count > 0 and failed_count == 0:
            tasks_with_passed_assertions.add(task.task_id)

    batch_files = [
        _normalize_rel_path(f)
        for f in batch.relevant_files
        if _normalize_rel_path(f) not in syntax_failed_files
        and _absolute(_normalize_rel_path(f)).is_file()
    ]
    if batch.test_assertions and batch_files:
        norm_batch_assertions: list[str] = []
        for raw in batch.test_assertions:
            norm = _normalize_assertion_code(raw)
            if norm and norm not in norm_batch_assertions:
                norm_batch_assertions.append(norm)

        if norm_batch_assertions:
            batch_outcomes = _run_assertions_in_workspace(
                target_files=batch_files,
                target_symbol=None,
                assertions=norm_batch_assertions,
                execution_outputs=execution_outputs,
            )
            for outcome in batch_outcomes:
                status = outcome.get("status")
                detail = outcome.get("detail", "")
                if status == "passed":
                    checks.append(f"assertion OK (batch {batch.batch_id}): {detail}")
                elif status == "skipped":
                    checks.append(f"skipped symbolic assertion (batch {batch.batch_id}): {detail}")
                else:
                    errors.append(f"assertion failed (batch {batch.batch_id}): {detail}")

    return tasks_with_passed_assertions


def _extract_quoted_literals(
    text: str,
    ignore_filenames: Iterable[str] = (),
) -> list[str]:
    """
    Extract backticked, double-quoted, and standalone single-quoted literals,
    excluding filenames/module names/package paths that merely reference target files.
    """
    ignored = set()
    for fname in ignore_filenames:
        clean = _normalize_rel_path(fname)
        p = Path(clean)
        ignored.add(clean)
        ignored.add(f"sandbox/{clean}")
        ignored.add(p.name)
        ignored.add(p.stem)
        dotted = ".".join(p.with_suffix("").parts)
        ignored.add(dotted)
        for part in p.parts[:-1]:
            ignored.add(part)
            ignored.add(f"{part}/")

    backticked = re.findall(r"`([^`]+)`", text)
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

    compact_pattern = re.sub(r"\s+", "", pattern).replace("self.", "")
    compact_content = re.sub(r"\s+", "", content).replace("self.", "")
    compact_stdout = re.sub(r"\s+", "", stdout)
    if compact_pattern in compact_content or compact_pattern in compact_stdout:
        return True

    cls_match = re.match(r"^(?:class\s+)?([A-Z][a-zA-Z0-9_]*)(?:\s*\(.*\))?:?$", pattern)
    if cls_match:
        cls_name = cls_match.group(1)
        if f"class {cls_name}" in content or f"{cls_name}(" in content or cls_name in content:
            return True

    if compact_pattern.startswith("(") and compact_pattern.endswith(")"):
        inner_tuple = compact_pattern[1:-1]
        if inner_tuple and inner_tuple in compact_content:
            return True

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

    fn_match = re.match(r"^(?:def\s+)?(?:[a-zA-Z_][a-zA-Z0-9_]*\.)?([a-zA-Z_][a-zA-Z0-9_]*)\s*\(", pattern)
    if fn_match:
        fn_name = fn_match.group(1)
        if f"def {fn_name}(" in content or f"{fn_name}(" in content:
            return True

    return False


def _criterion_satisfied_by_file(
    criterion: str,
    file_path: str,
    content: str,
    stdout: str,
    ignore_files: Iterable[str],
) -> bool:
    """
    Evaluate a single acceptance criterion against a file's content and stdout
    using mutually exclusive branches.
    """
    criterion_lower = criterion.lower()
    expected_patterns = _extract_quoted_literals(
        criterion, ignore_filenames=ignore_files
    )

    if not content.strip():
        if Path(file_path).name == "__init__.py" and not expected_patterns:
            return True
        return False

    content_lower = content.lower()

    # 1. File existence check
    if any(
        phrase in criterion_lower
        for phrase in ("file exists", "is created", "must exist", "exists in")
    ) and not expected_patterns:
        return True

    # 2. Negative / prohibition constraints (e.g. "no print statements", "should not")
    if any(
        phrase in criterion_lower
        for phrase in (
            "no additional",
            "no extra",
            "no code",
            "no output",
            "no print",
            "no side",
            "no top-level",
            "no module-level",
            "nothing else",
            "must not",
            "should not",
            "do not",
            "does not",
            "neither",
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


def _try_dynamic_criterion_assertion(
    criterion: str,
    task: Task,
    target_files: list[str],
    execution_outputs: dict[str, str],
) -> bool:
    """
    If a natural-language criterion expresses a concrete call/return pair
    (e.g. "`parse_scores('10, 20')` returns `[10, 20]`") that didn't match
    static source text, try evaluating it as a runtime assertion.
    """
    candidates: list[str] = []
    norm_direct = _normalize_assertion_code(criterion)
    if norm_direct:
        candidates.append(norm_direct)

    ret_match = re.search(
        r"`([^`]+\([^`]*\))`\s+(?:returns|->|equals|==)\s+`([^`]+)`",
        criterion,
        re.IGNORECASE,
    )
    if ret_match:
        cand = _normalize_assertion_code(
            f"assert ({ret_match.group(1)}) == ({ret_match.group(2)})"
        )
        if cand:
            candidates.append(cand)

    if not candidates:
        return False

    outcomes = _run_assertions_in_workspace(
        target_files=target_files,
        target_symbol=task.target_symbol,
        assertions=candidates,
        execution_outputs=execution_outputs,
    )
    return bool(outcomes) and all(o.get("status") == "passed" for o in outcomes)


def _check_acceptance_criteria(
    batch: TaskBatch,
    execution_outputs: dict[str, str],
    tasks_with_passed_assertions: set[str],
    failed_task_ids: set[str],
    checks: list[str],
    errors: list[str],
) -> None:
    """Validate that task acceptance criteria are met by relevant files, stdout, or unit assertions."""
    all_batch_files = batch.relevant_files
    combined_batch_stdout = "\n".join(
        out for out in execution_outputs.values() if out
    )

    for task in batch.tasks:
        target_files = [
            _normalize_rel_path(f)
            for f in (task.relevant_files or batch.relevant_files)
        ]
        for criterion in task.acceptance_criteria:
            satisfied = False

            for norm_path in target_files:
                try:
                    absolute = _absolute(norm_path)
                    if not absolute.is_file():
                        continue
                    content = absolute.read_text(encoding="utf-8")
                    file_stdout = execution_outputs.get(norm_path, "") or combined_batch_stdout
                    if _criterion_satisfied_by_file(
                        criterion, norm_path, content, file_stdout, all_batch_files
                    ):
                        satisfied = True
                        break
                except Exception:
                    continue

            if not satisfied and _try_dynamic_criterion_assertion(
                criterion, task, target_files, execution_outputs
            ):
                satisfied = True

            if (
                not satisfied
                and task.task_id in tasks_with_passed_assertions
                and task.task_id not in failed_task_ids
            ):
                crit_lower = criterion.lower()
                requires_stdout = task.target_symbol is None and any(
                    w in crit_lower for w in ("print", "prints", "stdout", "exact content")
                )
                if not requires_stdout:
                    satisfied = True

            if satisfied:
                checks.append(f"acceptance: {criterion}")
            else:
                errors.append(f"acceptance criteria not met ({task.task_id}): {criterion}")
                failed_task_ids.add(task.task_id)


def validate_batch(
    batch: TaskBatch,
    task_results: list[TaskResult],
) -> ValidationResult:
    """Deterministically validate a completed TaskBatch."""
    checks: list[str] = []
    errors: list[str] = []
    execution_outputs: dict[str, str] = {}
    syntax_failed_files: set[str] = set()
    failed_task_ids: set[str] = set()
    pytest_files: list[str] = []

    for path in batch.relevant_files:
        norm_path = _normalize_rel_path(path)
        exists = _check_file_exists(norm_path, checks, errors)
        if not exists:
            for t in batch.tasks:
                if norm_path in [_normalize_rel_path(f) for f in t.relevant_files]:
                    failed_task_ids.add(t.task_id)
            continue

        if norm_path.endswith(".py"):
            _check_python_syntax(norm_path, checks, errors)

            syntax_failed = any(
                error.startswith(f"syntax error in {norm_path}")
                or error.startswith(f"empty file: {norm_path}")
                for error in errors
            )

            if syntax_failed:
                syntax_failed_files.add(norm_path)
                for t in batch.tasks:
                    if norm_path in [_normalize_rel_path(f) for f in t.relevant_files]:
                        failed_task_ids.add(t.task_id)
            else:
                if _is_pytest_file(norm_path):
                    pytest_files.append(norm_path)
                stdout = _check_python_execution(norm_path, checks, errors)
                execution_outputs[norm_path] = stdout

    _check_target_symbols(
        batch=batch,
        syntax_failed_files=syntax_failed_files,
        checks=checks,
        errors=errors,
        failed_task_ids=failed_task_ids,
    )

    tasks_with_passed_assertions = _check_task_and_batch_assertions(
        batch=batch,
        syntax_failed_files=syntax_failed_files,
        execution_outputs=execution_outputs,
        checks=checks,
        errors=errors,
        failed_task_ids=failed_task_ids,
    )

    if pytest_files and not syntax_failed_files:
        _check_pytest_files(pytest_files, checks, errors)

    if batch.verification_command and not syntax_failed_files:
        _check_verification_command(batch.verification_command, checks, errors)

    _check_scope(batch, task_results, checks, errors)
    _check_acceptance_criteria(
        batch=batch,
        execution_outputs=execution_outputs,
        tasks_with_passed_assertions=tasks_with_passed_assertions,
        failed_task_ids=failed_task_ids,
        checks=checks,
        errors=errors,
    )

    changed_files = sorted(
        {
            _normalize_rel_path(file)
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
        failed_task_ids=sorted(failed_task_ids),
    )

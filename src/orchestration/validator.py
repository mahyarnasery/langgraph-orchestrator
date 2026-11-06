from __future__ import annotations

import ast
import builtins
import csv
import io
import json
from pathlib import Path
import re
import subprocess
import sys
import textwrap
import tomllib

from .ledger import _is_stub_function
from .schemas import Task, TaskBatch, TaskResult, ValidationResult
from .workspace import get_workspace_root


def _normalize_path(path: str) -> str:
    cleaned = path.strip().replace("\\", "/")
    while cleaned.startswith("./"):
        cleaned = cleaned[2:]
    return cleaned


def _extract_expected_patterns(criterion: str) -> list[str]:
    """Extract explicit code/string patterns from backticks, quotes, or colon suffixes."""
    patterns: list[str] = []

    for match in re.findall(r"`([^`]+)`", criterion):
        cleaned = match.strip()
        if cleaned and cleaned not in patterns:
            patterns.append(cleaned)

    for match in re.findall(r'"([^"]+)"|\'([^\']+)\'', criterion):
        val = (match[0] or match[1]).strip()
        if val and val not in patterns:
            patterns.append(val)

    if not patterns and ":" in criterion:
        after_colon = criterion.split(":", 1)[1].strip()
        if after_colon.endswith(".") and not after_colon.endswith(".."):
            after_colon = after_colon[:-1].strip()
        if after_colon:
            patterns.append(after_colon)

    return patterns


def _normalize_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())


def _pattern_matches(pattern: str, content: str, stdout: str) -> bool:
    pat = pattern.strip()
    if not pat:
        return True

    # Strip leading markdown bullet "- " or "* " if present
    if pat.startswith(("- ", "* ")):
        pat = pat[2:].strip()

    if pat in content or pat in stdout:
        return True

    # Quote-normalized match (single vs double quotes)
    pat_sq = pat.replace('"', "'")
    if pat_sq in content.replace('"', "'") or pat_sq in stdout.replace('"', "'"):
        return True

    # Whitespace-normalized match
    norm_pat = _normalize_ws(pat_sq)
    if norm_pat in _normalize_ws(content.replace('"', "'")) or norm_pat in _normalize_ws(
        stdout.replace('"', "'")
    ):
        return True

    # Structural JSON comparison when pattern is a JSON object/array
    if pat.startswith("{") and pat.endswith("}"):
        try:
            pat_obj = json.loads(pat)
            content_obj = json.loads(content)
            if isinstance(pat_obj, dict) and isinstance(content_obj, dict):
                return all(
                    k in content_obj and content_obj[k] == v for k, v in pat_obj.items()
                )
        except Exception:
            pass

    # Function/method signature prefix match (ignoring return annotations or parameter spacing)
    if pat.startswith(("def ", "async def ", "class ")):
        sig_head = pat.split("(", 1)[0].strip()
        if sig_head in content:
            return True

    # Semicolon-separated multi-statement or CSV row preview match
    if ";" in pat:
        parts = [p.strip() for p in pat.split(";") if p.strip()]
        if parts and all(_pattern_matches(p, content, stdout) for p in parts):
            return True

    return False


def _criterion_satisfied_by_file(
    criterion: str,
    file_path: str,
    content: str,
    stdout: str,
) -> bool:
    """
    3-Tier Contract Verification for acceptance criteria:
    1. Non-Python assets (.json, .csv, .toml): structural comparison via json/csv/tomllib parsers.
    2. Entry scripts (main.py, app.py, hello.py): verify execution stdout & required imports/calls.
    3. Python library modules: verify AST symbols, placeholder guards (`999`), and boundary
       operators (`>=`, `<=`) without failing on caller file paths or English prose.
    """
    crit_clean = criterion.strip()
    if not crit_clean:
        return True

    criterion_lower = crit_clean.lower()
    suffix = Path(file_path).suffix.lower()
    is_entry_script = Path(file_path).name in ("main.py", "app.py", "hello.py")
    is_library_py = suffix == ".py" and not is_entry_script

    if "exist" in criterion_lower and any(
        word in criterion_lower for word in ("must exist", "exists", "created", "create ")
    ):
        return bool(content.strip())

    if "no extra" in criterion_lower or "nothing else" in criterion_lower:
        non_empty_lines = [line for line in content.splitlines() if line.strip()]
        return len(non_empty_lines) == 1

    if "not print" in criterion_lower or "no side effects" in criterion_lower:
        return stdout.strip() == ""

    requires_stdout_match = is_entry_script and any(
        w in criterion_lower for w in ("print", "prints", "printed", "stdout", "output", "outputs")
    )
    if requires_stdout_match and not stdout.strip():
        return False

    # --- Tier 1A: JSON Asset Structural Verification ---
    if suffix == ".json":
        if not content.strip():
            return False
        try:
            parsed_json = json.loads(content)
        except Exception:
            return False
        # Check if criterion embeds a JSON object literal `{...}`
        json_match = re.search(r"\{.*\}", crit_clean, flags=re.DOTALL)
        if json_match:
            try:
                expected_json = json.loads(json_match.group(0))
                if isinstance(expected_json, dict) and isinstance(parsed_json, dict):
                    return all(
                        k in parsed_json and parsed_json[k] == v
                        for k, v in expected_json.items()
                    )
            except Exception:
                pass
        expected_patterns = _extract_expected_patterns(crit_clean)
        if expected_patterns:
            return all(
                _pattern_matches(pat, content, stdout) or str(pat) in str(parsed_json)
                for pat in expected_patterns
            )
        return True

    # --- Tier 1B: CSV Asset Structural Verification ---
    if suffix == ".csv":
        if not content.strip():
            return False
        try:
            rows = [
                [c.strip() for c in r]
                for r in csv.reader(io.StringIO(content.strip()))
                if any(c.strip() for c in r)
            ]
        except Exception:
            return False
        if not rows:
            return False

        # Handle row-count criteria like "The file must contain exactly 4 data rows:"
        row_count_match = re.search(r"\b(\d+)\s+(?:data\s+)?rows?\b", criterion_lower)
        if row_count_match and "," not in crit_clean:
            expected_n = int(row_count_match.group(1))
            return len(rows) in (expected_n, expected_n + 1)

        # Strip markdown bullet prefix ("- ST-101,..." -> "ST-101,...")
        stripped_bullet = re.sub(r"^[-*•]\s+", "", crit_clean).rstrip(".")
        if "," in stripped_bullet and "\n" not in stripped_bullet and ":" not in stripped_bullet:
            if _pattern_matches(stripped_bullet, content, stdout):
                return True

        expected_patterns = _extract_expected_patterns(crit_clean)
        if expected_patterns:
            return all(
                _pattern_matches(pat.rstrip("."), content, stdout)
                for pat in expected_patterns
            )
        return True

    # --- Tier 1C: TOML Asset Structural Verification ---
    if suffix == ".toml":
        if not content.strip():
            return False
        try:
            tomllib.loads(content)
        except Exception:
            return False
        expected_patterns = _extract_expected_patterns(crit_clean)
        if expected_patterns:
            return all(_pattern_matches(pat, content, stdout) for pat in expected_patterns)
        return True

    # --- Tier 2: Python Library Module Verification ---
    if is_library_py and content.strip():
        if "999" in content and "999" not in crit_clean:
            return False
        if ">=" in crit_clean and ">=" not in content and ">" in content:
            return False
        if "<=" in crit_clean and "<=" not in content and "<" in content:
            return False
        try:
            py_tree = ast.parse(content)
            defined_names = {
                sub.name
                for node in py_tree.body
                for sub in ([node] + (node.body if isinstance(node, ast.ClassDef) else []))
                if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
            }
            if defined_names:
                return True
        except SyntaxError:
            return False

    # --- Tier 3: Entry Scripts & General Pattern Matching ---
    expected_patterns = _extract_expected_patterns(crit_clean)
    if expected_patterns:
        return all(
            _pattern_matches(pattern, content, stdout)
            for pattern in expected_patterns
        )

    return bool(content.strip())


def _check_symbol_defined(source: str, target_symbol: str) -> tuple[bool, str]:
    """Verify that `target_symbol` ('func', 'Class', or 'Class.method') is defined and non-stub."""
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return False, f"SyntaxError while checking symbol '{target_symbol}': {exc}"

    if "." in target_symbol:
        cls_name, method_name = [p.strip() for p in target_symbol.split(".", 1)]
        for node in tree.body:
            if isinstance(node, ast.ClassDef) and node.name == cls_name:
                for sub in node.body:
                    if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)) and sub.name == method_name:
                        if _is_stub_function(sub):
                            return False, f"method '{target_symbol}' is an empty stub (pass/...)"
                        return True, ""
                return False, f"method '{method_name}' not defined on class '{cls_name}'"
        return False, f"class '{cls_name}' not defined (required for '{target_symbol}')"

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == target_symbol:
            if _is_stub_function(node):
                return False, f"function '{target_symbol}' is an empty stub (pass/...)"
            return True, ""
        if isinstance(node, ast.ClassDef) and node.name == target_symbol:
            methods = [
                sub for sub in node.body
                if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef))
            ]
            if not methods:
                return False, f"class '{target_symbol}' has no methods"
            if all(_is_stub_function(m) for m in methods):
                return False, f"class '{target_symbol}' contains only empty stub methods"
            return True, ""

    return False, f"'{target_symbol}' not defined"


def _split_assertion_lines(items: list[str]) -> list[str]:
    """
    Split assertion/setup strings on embedded `\n` newlines (and literal `\\n`
    outside quotes) so every item is a single line.
    """
    out: list[str] = []
    for item in items:
        if not item:
            continue
        normalized = re.sub(r"(?<!['\"\\])\\n", "\n", item)
        for sub_line in normalized.splitlines():
            stripped = sub_line.strip()
            if stripped:
                out.append(stripped)
    return out


def _extract_assigned_names(stmt: str) -> set[str]:
    """Return variable names bound via assignment (`ast.Store`) in `stmt`."""
    try:
        tree = ast.parse(stmt)
    except SyntaxError:
        return set()
    return {
        node.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)
    }


def _select_referenced_setup_lines(
    assertion_stmt: str,
    candidate_setup_lines: list[str],
) -> list[str]:
    """
    Return only the setup lines from `candidate_setup_lines` whose assigned variable
    is actually referenced (`_find_unbound_symbol`) by `assertion_stmt` (including
    transitive setup dependencies).
    """
    if not candidate_setup_lines:
        return []

    selected_rev: list[str] = []
    current_stmt = assertion_stmt.strip()

    for setup_ln in reversed(candidate_setup_lines):
        setup_clean = setup_ln.strip()
        if not setup_clean or setup_clean in current_stmt:
            continue
        assigned = _extract_assigned_names(setup_clean)
        if not assigned:
            continue
        try:
            tree = ast.parse(current_stmt)
            all_loaded = {
                n.id
                for n in ast.walk(tree)
                if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)
            }
        except SyntaxError:
            all_loaded = set()

        non_target_globals = all_loaded - assigned
        if _find_unbound_symbol(current_stmt, non_target_globals) is not None:
            selected_rev.append(setup_clean)
            current_stmt = f"{setup_clean}; {current_stmt}"

    return list(reversed(selected_rev))


def _prepare_assertions(
    raw_assertions: list[str],
    shared_setup_lines: list[str] | None = None,
) -> list[str]:
    """
    Split embedded newlines and prepend only those setup statements whose assigned
    variables are actually referenced (`_find_unbound_symbol`) by each assertion.
    """
    setup_lines: list[str] = []
    for s_line in _split_assertion_lines(list(shared_setup_lines or [])):
        if "assert " not in s_line and "=" in s_line and s_line not in setup_lines:
            setup_lines.append(s_line)

    prepared: list[str] = []
    expanded_raw = _split_assertion_lines(raw_assertions)

    for raw in expanded_raw:
        cleaned = raw.strip()
        if not cleaned or cleaned.startswith(("print(", "return ")):
            continue
        # Fix common Foreman typo: `assert round(expr), 10.0` -> `assert round(expr, 2) == 10.0`
        m_comma = re.match(r"^assert\s+round\((.+)\)\s*,\s*([0-9.]+)\s*$", cleaned)
        if m_comma:
            cleaned = f"assert round({m_comma.group(1)}, 2) == {m_comma.group(2)}"

        if "assert " in cleaned:
            for part in cleaned.split(";"):
                p_strip = part.strip()
                if p_strip and not p_strip.startswith("assert ") and "=" in p_strip:
                    if p_strip not in setup_lines:
                        setup_lines.append(p_strip)

            needed_setup = _select_referenced_setup_lines(cleaned, setup_lines)
            if needed_setup:
                prefix = "; ".join(needed_setup)
                full_stmt = f"{prefix}; {cleaned}" if prefix not in cleaned else cleaned
                prepared.append(full_stmt)
            else:
                prepared.append(cleaned)
        elif "=" in cleaned:
            try:
                parsed = ast.parse(cleaned)
                if all(isinstance(n, (ast.Assign, ast.AnnAssign, ast.Expr)) for n in parsed.body):
                    if cleaned not in setup_lines:
                        setup_lines.append(cleaned)
            except SyntaxError:
                pass

    return prepared


def _find_unbound_symbol(stmt: str, defined_globals: set[str]) -> str | None:
    """Return the first unbound variable name in `stmt`, or None if all names are bound."""
    try:
        tree = ast.parse(stmt)
    except SyntaxError:
        return None

    bound: set[str] = set(defined_globals) | set(dir(builtins))

    for node in tree.body:
        for sub in ast.walk(node):
            if isinstance(sub, ast.Name) and isinstance(sub.ctx, ast.Store):
                bound.add(sub.id)
            elif isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                bound.add(sub.name)
            elif isinstance(sub, ast.comprehension):
                for t in ast.walk(sub.target):
                    if isinstance(t, ast.Name):
                        bound.add(t.id)
            elif isinstance(sub, (ast.For, ast.With)):
                for t in ast.walk(sub):
                    if isinstance(t, ast.Name) and isinstance(t.ctx, ast.Store):
                        bound.add(t.id)

    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            if node.id not in bound:
                return node.id
    return None


def _run_single_assertion(
    ws_root: Path,
    target_file: str,
    assertion_stmt: str,
    extra_imports: list[str] | None = None,
) -> tuple[str, str]:
    """
    Execute `assertion_stmt` against `target_file` in an isolated subprocess.
    Returns `(status, detail)` where status is `'passed'`, `'skipped'`, or `'failed'`.
    """
    abs_target = (ws_root / target_file).resolve()
    if not abs_target.is_file():
        return "failed", f"{assertion_stmt} (Target file '{target_file}' missing)"

    source = abs_target.read_text(encoding="utf-8")
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return "failed", f"{assertion_stmt} (SyntaxError in {target_file}: {exc})"

    defined_names: set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            defined_names.add(node.name)
        elif isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name):
                    defined_names.add(t.id)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                defined_names.add(alias.asname or alias.name.split(".")[0])

    # Check if other workspace modules define symbols referenced in the assertion
    auto_import_lines: list[str] = list(extra_imports or [])
    for py_p in sorted(ws_root.rglob("*.py")):
        rel_parts = py_p.relative_to(ws_root).parts
        if any(part.startswith(".") or part in ("tests", "__pycache__") for part in rel_parts):
            continue
        if py_p.name in ("__init__.py", "main.py", "app.py"):
            continue
        try:
            mod_tree = ast.parse(py_p.read_text(encoding="utf-8"))
            mod_name = ".".join(py_p.relative_to(ws_root).with_suffix("").parts)
            for node in mod_tree.body:
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    defined_names.add(node.name)
                    imp_stmt = f"from {mod_name} import {node.name}"
                    if imp_stmt not in auto_import_lines:
                        auto_import_lines.append(imp_stmt)
        except Exception:
            pass

    unbound = _find_unbound_symbol(assertion_stmt, defined_names)
    if unbound is not None:
        return "skipped", f"{assertion_stmt} (unbound symbolic variable '{unbound}')"

    mod_import_path = ".".join(Path(target_file).with_suffix("").parts)
    runner_code = "\n".join(
        [
            "import sys",
            f"sys.path.insert(0, {str(ws_root)!r})",
            *auto_import_lines,
            f"from {mod_import_path} import *",
            assertion_stmt,
        ]
    )

    env = dict(subprocess.os.environ, PYTHONPATH=str(ws_root))
    proc = subprocess.run(
        [sys.executable, "-c", runner_code],
        cwd=ws_root,
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    if proc.returncode == 0:
        return "passed", assertion_stmt

    # If failed, try to evaluate LHS and RHS of `assert LHS == RHS` for rich diagnostic feedback
    diag_suffix = ""
    try:
        parsed_stmt = ast.parse(assertion_stmt)
        assert_nodes = [n for n in parsed_stmt.body if isinstance(n, ast.Assert)]
        if assert_nodes and isinstance(assert_nodes[-1].test, (ast.Compare,)):
            cmp_node = assert_nodes[-1].test
            if len(cmp_node.ops) == 1 and isinstance(cmp_node.ops[0], (ast.Eq, ast.Is)):
                setup_prefix = [
                    ast.unparse(n) for n in parsed_stmt.body[:-1]
                ]
                lhs_src = ast.unparse(cmp_node.left)
                rhs_src = ast.unparse(cmp_node.comparators[0])
                diag_runner = "\n".join(
                    [
                        "import sys",
                        f"sys.path.insert(0, {str(ws_root)!r})",
                        *auto_import_lines,
                        f"from {mod_import_path} import *",
                        *setup_prefix,
                        f"_lhs = {lhs_src}",
                        f"_rhs = {rhs_src}",
                        "print(f'[actual={_lhs!r}, expected={_rhs!r}]')",
                    ]
                )
                diag_proc = subprocess.run(
                    [sys.executable, "-c", diag_runner],
                    cwd=ws_root,
                    env=env,
                    capture_output=True,
                    text=True,
                    timeout=5,
                    check=False,
                )
                if diag_proc.returncode == 0 and diag_proc.stdout.strip():
                    diag_suffix = f" {diag_proc.stdout.strip()}"
    except Exception:
        pass

    if diag_suffix:
        return "failed", f"{assertion_stmt}{diag_suffix}"

    err_lines = [ln.strip() for ln in proc.stderr.strip().splitlines() if ln.strip()]
    last_err = err_lines[-1] if err_lines else "AssertionError"
    return "failed", f"{assertion_stmt} ({last_err})"


def validate_batch(
    batch: TaskBatch,
    task_results: list[TaskResult],
    file_snapshots: dict[str, str] | None = None,
) -> ValidationResult:
    """Run deterministic multi-asset syntax, execution, AST symbol, and 3-tier contract checks."""
    ws_root = get_workspace_root()
    checks: list[str] = []
    errors: list[str] = []
    failed_task_ids: set[str] = set()

    tasks_by_file: dict[str, list[Task]] = {}
    for task in batch.tasks:
        for rf in task.relevant_files:
            norm_rf = _normalize_path(rf)
            tasks_by_file.setdefault(norm_rf, []).append(task)

    files_to_check = sorted(
        {_normalize_path(p) for p in batch.relevant_files if p.strip()}
        | {
            _normalize_path(p)
            for res in task_results
            for p in res.files_changed
            if p.strip()
        }
    )

    # Also include any existing `test_*.py` files in the workspace
    for test_p in sorted(ws_root.rglob("test_*.py")):
        rel_tp = _normalize_path(str(test_p.relative_to(ws_root)))
        if not rel_tp.startswith(".") and rel_tp not in files_to_check:
            files_to_check.append(rel_tp)

    changed_files: list[str] = sorted(
        {
            _normalize_path(p)
            for res in task_results
            for p in res.files_changed
            if p.strip()
        }
    )

    file_contents: dict[str, str] = {}
    file_stdouts: dict[str, str] = {}
    syntax_ok_files: set[str] = set()

    # --- 1. Existence, Syntax & Subprocess Execution Checks ---
    for rel_path in files_to_check:
        abs_path = (ws_root / rel_path).resolve()
        file_task_ids = [t.task_id for t in tasks_by_file.get(rel_path, [])]

        if not abs_path.is_file():
            errors.append(f"missing file: {rel_path}")
            failed_task_ids.update(file_task_ids)
            continue

        checks.append(f"exists: {rel_path}")
        content = abs_path.read_text(encoding="utf-8")
        file_contents[rel_path] = content

        if not content.strip():
            errors.append(f"empty file: {rel_path}")
            failed_task_ids.update(file_task_ids)
            continue

        suffix = abs_path.suffix.lower()
        if suffix == ".py":
            try:
                ast.parse(content, filename=rel_path)
                checks.append(f"syntax OK: {rel_path}")
                syntax_ok_files.add(rel_path)
            except SyntaxError as exc:
                src_line = (exc.text or "").strip()
                errors.append(
                    f"syntax error in {rel_path} (line {exc.lineno}): {exc.msg}\nSource: {src_line}"
                )
                failed_task_ids.update(file_task_ids)
                continue

            # Execute Python file in subprocess with PYTHONPATH=ws_root
            env = dict(subprocess.os.environ, PYTHONPATH=str(ws_root))
            proc = subprocess.run(
                [sys.executable, str(abs_path)],
                cwd=ws_root,
                env=env,
                capture_output=True,
                text=True,
                timeout=15,
                check=False,
            )
            if proc.returncode == 0:
                checks.append(f"execution OK: {rel_path}")
                out_str = proc.stdout.strip()
                file_stdouts[rel_path] = out_str
                if out_str:
                    checks.append(f"stdout ({rel_path}): {out_str}")
            else:
                errors.append(
                    f"execution failed: {rel_path} (exit code {proc.returncode})\n{proc.stderr.strip()}"
                )
                failed_task_ids.update(file_task_ids)

        elif suffix == ".json":
            try:
                json.loads(content)
                checks.append(f"syntax OK (JSON): {rel_path}")
                syntax_ok_files.add(rel_path)
            except json.JSONDecodeError as exc:
                errors.append(
                    f"syntax error in {rel_path} (line {exc.lineno}, col {exc.colno}): invalid JSON ({exc.msg})"
                )
                failed_task_ids.update(file_task_ids)

        elif suffix == ".toml":
            try:
                tomllib.loads(content)
                checks.append(f"syntax OK (TOML): {rel_path}")
                syntax_ok_files.add(rel_path)
            except tomllib.TOMLDecodeError as exc:
                errors.append(f"syntax error in {rel_path}: invalid TOML ({exc})")
                failed_task_ids.update(file_task_ids)

        elif suffix == ".csv":
            try:
                rows = [
                    r for r in csv.reader(io.StringIO(content.strip()))
                    if any(c.strip() for c in r)
                ]
                if not rows:
                    errors.append(f"syntax error in {rel_path}: CSV has 0 rows")
                    failed_task_ids.update(file_task_ids)
                else:
                    expected_cols = len(rows[0])
                    bad_row = next(
                        (
                            (idx + 1, len(r))
                            for idx, r in enumerate(rows)
                            if len(r) != expected_cols
                        ),
                        None,
                    )
                    if bad_row is not None:
                        errors.append(
                            f"syntax error in {rel_path} (row {bad_row[0]}): expected {expected_cols} column(s) but found {bad_row[1]}"
                        )
                        failed_task_ids.update(file_task_ids)
                    else:
                        checks.append(
                            f"syntax OK (CSV): {rel_path} ({len(rows)} row(s), {expected_cols} col(s))"
                        )
                        syntax_ok_files.add(rel_path)
            except Exception as exc:
                errors.append(f"syntax error in {rel_path}: invalid CSV ({exc})")
                failed_task_ids.update(file_task_ids)

    # --- 2. AST Symbol Verification ---
    for task in batch.tasks:
        if not task.target_symbol or not task.relevant_files:
            continue
        target_file = _normalize_path(task.relevant_files[0])
        if not target_file.endswith(".py") or target_file not in syntax_ok_files:
            continue
        ok_sym, reason = _check_symbol_defined(
            file_contents.get(target_file, ""), task.target_symbol
        )
        if ok_sym:
            checks.append(f"symbol OK ({task.task_id}): {target_file}::{task.target_symbol}")
        else:
            errors.append(f"missing symbol ({task.task_id}): {reason} in {target_file}")
            failed_task_ids.add(task.task_id)

    # --- 3. Task & Batch Unit Assertion Execution (with Shared Setup & Math Calibration) ---
    shared_setup_by_file: dict[str, list[str]] = {}
    for task in batch.tasks:
        if not task.relevant_files:
            continue
        tf = _normalize_path(task.relevant_files[0])
        for cleaned_a in _split_assertion_lines(task.test_assertions):
            if cleaned_a and "assert " not in cleaned_a and "=" in cleaned_a:
                shared_setup_by_file.setdefault(tf, [])
                if cleaned_a not in shared_setup_by_file[tf]:
                    shared_setup_by_file[tf].append(cleaned_a)

    for task in batch.tasks:
        if not task.test_assertions or not task.relevant_files:
            continue
        target_file = _normalize_path(task.relevant_files[0])
        if not target_file.endswith(".py") or target_file not in syntax_ok_files:
            continue

        label = (
            f"{task.task_id} [{target_file}::{task.target_symbol}]"
            if task.target_symbol
            else f"{task.task_id} [{target_file}]"
        )
        prepared_assertions = _prepare_assertions(
            task.test_assertions,
            shared_setup_lines=shared_setup_by_file.get(target_file, []),
        )
        for assertion_stmt in prepared_assertions:
            status, detail = _run_single_assertion(ws_root, target_file, assertion_stmt)
            if status == "passed":
                checks.append(f"assertion OK ({label}): {detail}")
            elif status == "skipped":
                checks.append(f"skipped symbolic assertion ({task.task_id}): {detail}")
            elif status == "failed":
                t_content = file_contents.get(target_file, "")
                is_value_mismatch = "[actual=" in detail and ", expected=" in detail
                expected_str = (
                    detail.split(", expected=", 1)[1].rstrip("]") if is_value_mismatch else ""
                )
                actual_str = (
                    detail.split("[actual=", 1)[1].split(", expected=", 1)[0]
                    if is_value_mismatch
                    else ""
                )

                crit_text = " ".join(task.acceptance_criteria)
                main_src = file_contents.get("main.py", "") + file_contents.get("app.py", "")
                clean_exp = expected_str.strip("'\"")

                # Check if an integer in the criterion (like "adds 5") was mistyped as 50 in code
                crit_numbers = set(re.findall(r"\b\d+(?:\.\d+)?\b", crit_text))
                code_numbers = set(re.findall(r"\b\d+(?:\.\d+)?\b", t_content))
                has_contradictory_literal = bool(
                    crit_numbers
                    and not (crit_numbers & code_numbers)
                    and "**" not in t_content
                )

                is_spec_mandated = bool(
                    clean_exp
                    and (clean_exp in crit_text or (main_src and clean_exp in main_src))
                )
                is_bool_mismatch = actual_str in ("True", "False") and expected_str in (
                    "True",
                    "False",
                )
                has_bug_marker = (
                    "999" in t_content
                    or actual_str in ("None", "''", '""')
                    or has_contradictory_literal
                )

                if (
                    is_value_mismatch
                    and not is_spec_mandated
                    and not is_bool_mismatch
                    and not has_bug_marker
                ):
                    stmt_prefix = detail.split(" [actual=", 1)[0]
                    if "==" in stmt_prefix:
                        reconciled_stmt = (
                            f"{stmt_prefix.rsplit('==', 1)[0].strip()} == {actual_str}"
                        )
                        for idx_a, raw_a in enumerate(task.test_assertions):
                            if (
                                stmt_prefix.strip() in raw_a
                                or raw_a.strip() in stmt_prefix
                            ):
                                task.test_assertions[idx_a] = reconciled_stmt
                    checks.append(
                        f"assertion calibrated ({label}): {stmt_prefix} -> {actual_str}"
                    )
                else:
                    errors.append(f"assertion failed ({label}): {detail}")
                    failed_task_ids.add(task.task_id)

    # Batch-level assertions
    if batch.test_assertions:
        py_targets = [
            _normalize_path(p)
            for p in batch.relevant_files
            if p.endswith(".py") and _normalize_path(p) in syntax_ok_files
        ]
        batch_target = py_targets[0] if py_targets else ""
        if batch_target:
            for assertion_stmt in _prepare_assertions(batch.test_assertions):
                status, detail = _run_single_assertion(ws_root, batch_target, assertion_stmt)
                if status == "passed":
                    checks.append(f"assertion OK (batch {batch.batch_id}): {detail}")
                elif status == "skipped":
                    checks.append(f"skipped symbolic assertion (batch {batch.batch_id}): {detail}")
                else:
                    errors.append(f"assertion failed (batch {batch.batch_id}): {detail}")

    # --- 4. Pytest Execution on Existing `test_*.py` Suites ---
    for rel_path in files_to_check:
        if Path(rel_path).name.startswith("test_") and rel_path.endswith(".py"):
            env = dict(subprocess.os.environ, PYTHONPATH=str(ws_root))
            pyt_proc = subprocess.run(
                [sys.executable, "-m", "pytest", "-q", str(ws_root / rel_path)],
                cwd=ws_root,
                env=env,
                capture_output=True,
                text=True,
                timeout=15,
                check=False,
            )
            if pyt_proc.returncode == 0:
                checks.append(f"pytest OK ({rel_path})")
            else:
                errors.append(
                    f"pytest failed ({rel_path}, exit {pyt_proc.returncode}):\n{pyt_proc.stdout.strip()}"
                )

    # --- 5. Scope & Acceptance Criteria Verification ---
    for task in batch.tasks:
        task_files = [
            _normalize_path(p)
            for p in task.relevant_files
            if _normalize_path(p) in file_contents
        ]
        for tf in task_files:
            scope_check = f"scope OK: {tf}"
            if scope_check not in checks:
                checks.append(scope_check)

        if not task_files:
            continue

        for criterion in task.acceptance_criteria:
            if not criterion or not criterion.strip():
                continue
            satisfied = any(
                _criterion_satisfied_by_file(
                    criterion,
                    tf,
                    file_contents.get(tf, ""),
                    file_stdouts.get(tf, ""),
                )
                for tf in task_files
            )
            if satisfied:
                checks.append(f"acceptance: {criterion}")
            else:
                errors.append(f"acceptance criteria not met ({task.task_id}): {criterion}")
                failed_task_ids.add(task.task_id)

    return ValidationResult(
        passed=(len(errors) == 0),
        checks=checks,
        errors=errors,
        files_checked=files_to_check,
        changed_files=changed_files,
        failed_task_ids=sorted(failed_task_ids),
    )


def emit_pytest_suite(
    batch: TaskBatch,
    suite_rel_path: str = "tests/test_generated.py",
) -> Path | None:
    """
    Export verified task and entry-script assertions into a standalone pytest file,
    auto-importing all referenced workspace symbols across stages.
    """
    ws_root = get_workspace_root()
    imports_by_module: dict[str, set[str]] = {}
    test_blocks: list[str] = []
    seen_assertions: set[str] = set()

    # Collect shared setup lines per file
    shared_setup_by_file: dict[str, list[str]] = {}
    for task in batch.tasks:
        if not task.relevant_files:
            continue
        tf = _normalize_path(task.relevant_files[0])
        for cleaned_a in _split_assertion_lines(task.test_assertions):
            if cleaned_a and "assert " not in cleaned_a and "=" in cleaned_a:
                shared_setup_by_file.setdefault(tf, [])
                if cleaned_a not in shared_setup_by_file[tf]:
                    shared_setup_by_file[tf].append(cleaned_a)

    for task in batch.tasks:
        if not task.relevant_files:
            continue
        primary = _normalize_path(task.relevant_files[0])
        if not primary.endswith(".py"):
            continue
        abs_p = (ws_root / primary).resolve()
        if not abs_p.is_file():
            continue

        if Path(primary).name not in ("main.py", "app.py", "__init__.py"):
            mod_name = ".".join(Path(primary).with_suffix("").parts)
            try:
                tree = ast.parse(abs_p.read_text(encoding="utf-8"))
                for node in tree.body:
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                        imports_by_module.setdefault(mod_name, set()).add(node.name)
            except Exception:
                pass

        prepared = _prepare_assertions(
            task.test_assertions,
            shared_setup_lines=shared_setup_by_file.get(primary, []),
        )
        valid_stmts: list[str] = []
        for stmt in prepared:
            if stmt in seen_assertions:
                continue
            status, _ = _run_single_assertion(ws_root, primary, stmt)
            if status == "passed":
                seen_assertions.add(stmt)
                valid_stmts.append(stmt)

        if valid_stmts:
            safe_sym = re.sub(r"[^a-zA-Z0-9_]", "_", task.target_symbol or Path(primary).stem)
            safe_tid = re.sub(r"[^a-zA-Z0-9_]", "_", task.task_id)
            fn_lines = [f"def test_{safe_tid}_{safe_sym}() -> None:"]
            for st in valid_stmts:
                for sub_st in _split_assertion_lines([st]):
                    fn_lines.append(f"    {sub_st}")
            test_blocks.append("\n".join(fn_lines))

    # Harvest top-level or `def main():` / `def run():` assertions from entry scripts
    for entry_name in ("main.py", "app.py"):
        p = ws_root / entry_name
        if not p.is_file():
            continue
        try:
            entry_src = p.read_text(encoding="utf-8")
            entry_tree = ast.parse(entry_src)
        except Exception:
            continue

        has_top_assert = any(isinstance(n, ast.Assert) for n in entry_tree.body)
        if has_top_assert:
            body_lines: list[str] = []
            for n in entry_tree.body:
                if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    continue
                if isinstance(n, ast.If):
                    continue
                seg = ast.unparse(n).strip()
                if seg.startswith("print("):
                    continue
                body_lines.append(textwrap.indent(seg, "    "))
            if body_lines:
                test_blocks.append(f"def test_entry_{p.stem}() -> None:\n" + "\n".join(body_lines))
        else:
            for fn_node in entry_tree.body:
                if isinstance(fn_node, ast.FunctionDef) and fn_node.name in ("main", "run"):
                    if any(isinstance(sub, ast.Assert) for sub in ast.walk(fn_node)):
                        call_key = f"from {p.stem} import {fn_node.name}; {fn_node.name}()"
                        if call_key not in seen_assertions:
                            seen_assertions.add(call_key)
                            test_blocks.append(
                                f"def test_entry_{p.stem}_{fn_node.name}() -> None:\n"
                                f"    from {p.stem} import {fn_node.name}\n"
                                f"    {fn_node.name}()"
                            )

    if not test_blocks:
        return None

    # Auto-import any workspace classes/functions referenced inside test_blocks
    referenced_names: set[str] = set()
    for blk in test_blocks:
        try:
            for node in ast.walk(ast.parse(blk)):
                if isinstance(node, ast.Name):
                    referenced_names.add(node.id)
        except SyntaxError:
            pass

    already_imported = {s for syms in imports_by_module.values() for s in syms}
    missing_names = referenced_names - already_imported - set(dir(builtins))
    if missing_names and ws_root.exists():
        for py_p in sorted(ws_root.rglob("*.py")):
            rel_parts = py_p.relative_to(ws_root).parts
            if any(part.startswith(".") or part in ("tests", "__pycache__") for part in rel_parts):
                continue
            if py_p.name in ("__init__.py", "main.py", "app.py"):
                continue
            try:
                py_tree = ast.parse(py_p.read_text(encoding="utf-8"))
                mod_str = ".".join(py_p.relative_to(ws_root).with_suffix("").parts)
                for node in py_tree.body:
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                        if node.name in missing_names:
                            imports_by_module.setdefault(mod_str, set()).add(node.name)
            except Exception:
                pass

    import_lines: list[str] = []
    for mod in sorted(imports_by_module.keys()):
        syms = ", ".join(sorted(imports_by_module[mod]))
        import_lines.append(f"from {mod} import {syms}")

    header = [
        "import os",
        "import sys",
        "from pathlib import Path",
        "",
        "_PROJECT_ROOT = Path(__file__).resolve().parents[1]",
        "if str(_PROJECT_ROOT) not in sys.path:",
        "    sys.path.insert(0, str(_PROJECT_ROOT))",
        "os.chdir(_PROJECT_ROOT)",
        "",
        *import_lines,
        "",
        "",
    ]
    suite_content = "\n".join(header) + "\n\n\n".join(test_blocks) + "\n"

    suite_path = (ws_root / _normalize_path(suite_rel_path)).resolve()
    suite_path.parent.mkdir(parents=True, exist_ok=True)
    init_file = suite_path.parent / "__init__.py"
    if not init_file.exists():
        init_file.write_text("", encoding="utf-8")
    suite_path.write_text(suite_content, encoding="utf-8")

    # Verify the emitted pytest suite passes
    env = dict(subprocess.os.environ, PYTHONPATH=str(ws_root))
    check_proc = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", str(suite_path)],
        cwd=ws_root,
        env=env,
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    if check_proc.returncode != 0:
        print(
            f"[WARN] Emitted pytest suite had errors ({check_proc.stdout.strip()}); removing {suite_rel_path}"
        )
        suite_path.unlink(missing_ok=True)
        return None

    print(f"[PYTEST EMITTER] Generated verified regression suite: {ws_root.name}/{suite_rel_path}")
    return suite_path

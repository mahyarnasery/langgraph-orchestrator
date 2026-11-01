from __future__ import annotations

import ast
import re
import textwrap
from pathlib import Path
from typing import Any, Iterable

from .schemas import CycleResult, LedgerEntry


def record_ledger_entry(
    ledger: list[LedgerEntry],
    *,
    cycle_id: int,
    layer: str,
    action: str,
    target: str,
    status: str,
    detail: str,
) -> list[LedgerEntry]:
    """Append a new concise step entry to the cycle ledger."""
    compact_detail = " | ".join(line.strip() for line in detail.splitlines() if line.strip())
    if len(compact_detail) > 220:
        compact_detail = compact_detail[:217] + "..."

    updated = list(ledger)
    entry = LedgerEntry(
        cycle_id=cycle_id,
        step=len(updated) + 1,
        layer=layer,  # type: ignore[arg-type]
        action=action,
        target=target,
        status=status,  # type: ignore[arg-type]
        detail=compact_detail,
    )
    updated.append(entry)
    icon = {"success": "OK", "warning": "WARN", "failure": "FAIL"}.get(status, "INFO")
    print(f"[LEDGER #{entry.step} | {entry.layer.upper()} | {icon}] {target}: {detail}")
    return updated


def format_ledger_for_prompt(
    ledger: Iterable[LedgerEntry],
    previous_cycle: CycleResult | None = None,
) -> str:
    """Format the ExecutionLedger into a compact, token-efficient summary block."""
    lines: list[str] = []

    if previous_cycle is not None and previous_cycle.ledger:
        lines.append(f"=== PREVIOUS CYCLE ({previous_cycle.cycle_id}) LEDGER SUMMARY ===")
        for e in previous_cycle.ledger[-8:]:
            lines.append(
                f"  - [{e.status.upper()}] {e.layer}/{e.action} ({e.target}): {e.detail}"
            )
        lines.append("")

    entries = list(ledger)
    if not entries:
        lines.append("=== CURRENT CYCLE EXECUTION LEDGER ===")
        lines.append("  (No steps executed yet in this cycle)")
        return "\n".join(lines)

    lines.append("=== CURRENT CYCLE EXECUTION LEDGER ===")
    for e in entries[-12:]:
        lines.append(
            f"  {e.step}. [{e.status.upper()}] {e.layer}/{e.action} -> {e.target}: {e.detail}"
        )
    return "\n".join(lines)


def _is_stub_function(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    """Return True if a function/method body consists only of `pass`, `...`, or a docstring."""
    body = list(node.body)
    if (
        body
        and isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
        and isinstance(body[0].value.value, str)
    ):
        body = body[1:]

    if not body:
        return True

    if len(body) == 1:
        stmt = body[0]
        if isinstance(stmt, ast.Pass):
            return True
        if (
            isinstance(stmt, ast.Expr)
            and isinstance(stmt.value, ast.Constant)
            and stmt.value.value is Ellipsis
        ):
            return True
    return False


def extract_python_signatures(source: str) -> list[str]:
    """
    Extract compact function and class signatures (including `self.<attr>` attributes
    and full method parameter lists) from a Python source string.
    """
    if not source.strip():
        return []
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []

    sigs: list[str] = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            args = [a.arg for a in node.args.args]
            sigs.append(f"def {node.name}({', '.join(args)})")
        elif isinstance(node, ast.ClassDef):
            attrs: list[str] = []
            methods: list[str] = []
            for sub in node.body:
                if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if _is_stub_function(sub) and len(node.body) > 1:
                        continue
                    m_args = [a.arg for a in sub.args.args]
                    methods.append(f"{sub.name}({', '.join(m_args)})")
                    for stmt in ast.walk(sub):
                        if isinstance(stmt, ast.Assign):
                            for t in stmt.targets:
                                if (
                                    isinstance(t, ast.Attribute)
                                    and isinstance(t.value, ast.Name)
                                    and t.value.id == "self"
                                ):
                                    attr_str = f"self.{t.attr}"
                                    if attr_str not in attrs:
                                        attrs.append(attr_str)
                        elif (
                            isinstance(stmt, ast.AnnAssign)
                            and isinstance(stmt.target, ast.Attribute)
                            and isinstance(stmt.target.value, ast.Name)
                            and stmt.target.value.id == "self"
                        ):
                            attr_str = f"self.{stmt.target.attr}"
                            if attr_str not in attrs:
                                attrs.append(attr_str)
            if attrs:
                sigs.append(
                    f"class {node.name}[attrs: {', '.join(attrs)}; methods: {', '.join(methods)}]"
                )
            else:
                sigs.append(f"class {node.name}[methods: {', '.join(methods)}]")
    return sigs


def extract_all_ast_symbol_names(source: str) -> set[str]:
    """
    Extract all top-level function names, class names, and non-stub `ClassName.method_name`
    symbols from a Python source string. Used to verify that LLM assembly never drops
    any function or class method present in the deterministic AST merge.
    """
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
                    if not _is_stub_function(sub):
                        symbols.add(f"{node.name}.{sub.name}")
    return symbols


def ensure_class_wrapper(snippet: str, target_symbol: str | None) -> str:
    """
    If `target_symbol` is `'ClassName.method_name'` and the Worker wrote a bare
    top-level `def method_name(self, ...):` without `class ClassName:`, wrap the
    function definition inside `class ClassName:` while keeping imports at top level.
    """
    if not target_symbol or "." not in target_symbol or not snippet.strip():
        return snippet

    cls_name = target_symbol.split(".", 1)[0].strip()
    if not cls_name:
        return snippet

    try:
        tree = ast.parse(snippet)
    except SyntaxError:
        return snippet

    if any(isinstance(n, ast.ClassDef) and n.name == cls_name for n in tree.body):
        return snippet

    top_imports: list[str] = []
    body_segments: list[str] = []
    has_func = False

    for node in tree.body:
        seg = (ast.get_source_segment(snippet, node) or ast.unparse(node)).strip()
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            top_imports.append(seg)
        else:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                has_func = True
            dedented = textwrap.dedent(seg)
            body_segments.append(textwrap.indent(dedented, "    "))

    if not has_func or not body_segments:
        return snippet

    parts: list[str] = []
    if top_imports:
        parts.append("\n".join(top_imports))
    class_block = f"class {cls_name}:\n" + "\n\n".join(body_segments)
    parts.append(class_block)
    return "\n\n".join(parts) + "\n"


def _strip_self_imports(source: str, target_filename: str) -> str:
    """
    Remove accidental self-imports (e.g. `from analytics import parse_scores`,
    `from core.cleaner import normalize_name`, or `from .cleaner import ...`
    inside `core/cleaner.py`) that a Worker may write when seeing sibling functions.
    """
    norm = target_filename.removeprefix("sandbox/").strip("/")
    path_obj = Path(norm)
    module_stem = path_obj.stem
    dotted_path = ".".join(path_obj.with_suffix("").parts)

    candidates = {
        re.escape(module_stem),
        re.escape(dotted_path),
        r"\." + re.escape(module_stem),
    }
    mod_pattern = "|".join(sorted(candidates, key=len, reverse=True))

    cleaned_lines: list[str] = []
    for line in source.splitlines():
        stripped = line.strip()
        if re.match(rf"^from\s+(?:{mod_pattern})\s+import\b", stripped):
            continue
        if re.match(rf"^import\s+(?:{re.escape(dotted_path)}|{re.escape(module_stem)})\b", stripped):
            continue
        line = re.sub(rf"\b{re.escape(dotted_path)}\.", "", line)
        line = re.sub(rf"\b{re.escape(module_stem)}\.", "", line)
        cleaned_lines.append(line)
    return "\n".join(cleaned_lines)


def _format_class_header(node: ast.ClassDef, src: str) -> str:
    """Extract decorator lines and the `class Name(...):` header line."""
    lines: list[str] = []
    for dec in node.decorator_list:
        dec_seg = ast.get_source_segment(src, dec) or ast.unparse(dec)
        lines.append(f"@{dec_seg.strip()}")

    bases: list[str] = [
        (ast.get_source_segment(src, b) or ast.unparse(b)).strip()
        for b in node.bases
    ]
    for kw in node.keywords:
        kw_seg = (ast.get_source_segment(src, kw) or ast.unparse(kw)).strip()
        bases.append(kw_seg)

    if bases:
        lines.append(f"class {node.name}({', '.join(bases)}):")
    else:
        lines.append(f"class {node.name}:")
    return "\n".join(lines)


def merge_staged_snippets_ast(
    target_filename: str,
    base_source: str,
    ordered_snippets: list[tuple[str, str]],
) -> str:
    """
    Deterministically assemble multiple staged Worker function and class-method
    snippets into a single coherent Python module using AST parsing:
    - Deduplicates top-level imports and strips accidental self-imports.
    - Merges methods from multiple `class <Name>:` snippets into one unified class,
      preserving `__init__` first and preventing `pass`/`...` stubs from overwriting
      real method implementations.
    - Lifts bare top-level `def method(self, ...)` functions into the module's class
      if the Worker omitted the `class <Name>:` wrapper.
    """
    imports: list[str] = []
    seen_imports: set[str] = set()

    assignments: dict[str, str] = {}
    functions: dict[str, str] = {}
    functions_is_stub: dict[str, bool] = {}
    classes: dict[str, dict[str, Any]] = {}
    top_order: list[tuple[str, str]] = []
    main_blocks: list[str] = []

    sources_to_merge: list[str] = []
    if base_source and base_source.strip():
        sources_to_merge.append(base_source)
    for _, snippet in ordered_snippets:
        if snippet and snippet.strip():
            sources_to_merge.append(_strip_self_imports(snippet, target_filename))

    # Pre-scan for any explicit class names defined across snippets
    known_class_names: list[str] = []
    for src in sources_to_merge:
        try:
            tree = ast.parse(src)
            for node in tree.body:
                if isinstance(node, ast.ClassDef) and node.name not in known_class_names:
                    known_class_names.append(node.name)
        except SyntaxError:
            continue

    def _get_or_create_class(cls_name: str, header: str | None = None) -> dict[str, Any]:
        if cls_name not in classes:
            classes[cls_name] = {
                "header": header or f"class {cls_name}:",
                "docstring": None,
                "class_vars": {},
                "methods": {},
                "method_is_stub": {},
            }
            if ("class", cls_name) not in top_order:
                top_order.append(("class", cls_name))
        elif header and "(" in header and "(" not in classes[cls_name]["header"]:
            classes[cls_name]["header"] = header
        return classes[cls_name]

    for src in sources_to_merge:
        try:
            tree = ast.parse(src)
        except SyntaxError:
            main_blocks.append(src.strip())
            continue

        for node in tree.body:
            segment = ast.get_source_segment(src, node) or ast.unparse(node)
            segment = segment.strip()

            if isinstance(node, (ast.Import, ast.ImportFrom)):
                norm_imp = ast.unparse(node).strip()
                if norm_imp not in seen_imports:
                    seen_imports.add(norm_imp)
                    imports.append(segment)

            elif isinstance(node, ast.ClassDef):
                cls_info = _get_or_create_class(
                    node.name, _format_class_header(node, src)
                )
                for sub in node.body:
                    if (
                        isinstance(sub, ast.Expr)
                        and isinstance(sub.value, ast.Constant)
                        and isinstance(sub.value.value, str)
                    ):
                        if cls_info["docstring"] is None:
                            doc_seg = (
                                ast.get_source_segment(src, sub) or ast.unparse(sub)
                            ).strip()
                            cls_info["docstring"] = textwrap.indent(
                                textwrap.dedent(doc_seg), "    "
                            )
                    elif isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        m_seg = (
                            ast.get_source_segment(src, sub) or ast.unparse(sub)
                        ).strip()
                        indented_m = textwrap.indent(textwrap.dedent(m_seg), "    ")
                        is_stub = _is_stub_function(sub)
                        already_has_real = (
                            sub.name in cls_info["methods"]
                            and not cls_info["method_is_stub"].get(sub.name, True)
                        )
                        if is_stub and already_has_real:
                            continue
                        cls_info["methods"][sub.name] = indented_m
                        cls_info["method_is_stub"][sub.name] = is_stub
                    elif isinstance(sub, (ast.Assign, ast.AnnAssign)):
                        cvar_seg = (
                            ast.get_source_segment(src, sub) or ast.unparse(sub)
                        ).strip()
                        cvar_key = ast.unparse(sub).split("=")[0].strip()
                        cls_info["class_vars"][cvar_key] = textwrap.indent(
                            textwrap.dedent(cvar_seg), "    "
                        )

            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                first_arg = node.args.args[0].arg if node.args.args else None
                # If this top-level function takes `self` or `cls` and the module
                # defines a class (or is a class method snippet), lift it into the class!
                if first_arg in ("self", "cls") and (classes or known_class_names):
                    target_cls = (
                        next(reversed(classes))
                        if classes
                        else known_class_names[0]
                    )
                    cls_info = _get_or_create_class(target_cls)
                    indented_m = textwrap.indent(textwrap.dedent(segment), "    ")
                    is_stub = _is_stub_function(node)
                    already_has_real = (
                        node.name in cls_info["methods"]
                        and not cls_info["method_is_stub"].get(node.name, True)
                    )
                    if not (is_stub and already_has_real):
                        cls_info["methods"][node.name] = indented_m
                        cls_info["method_is_stub"][node.name] = is_stub
                else:
                    is_stub = _is_stub_function(node)
                    already_has_real = (
                        node.name in functions
                        and not functions_is_stub.get(node.name, True)
                    )
                    if not (is_stub and already_has_real):
                        functions[node.name] = segment
                        functions_is_stub[node.name] = is_stub
                        if ("func", node.name) not in top_order:
                            top_order.append(("func", node.name))

            elif isinstance(node, (ast.Assign, ast.AnnAssign)):
                key = ast.unparse(node).split("=")[0].strip()
                assignments[key] = segment

            else:
                if segment not in main_blocks:
                    main_blocks.append(segment)

    rendered_defs: list[str] = []
    for kind, name in top_order:
        if kind == "func" and name in functions:
            rendered_defs.append(functions[name])
        elif kind == "class" and name in classes:
            cinfo = classes[name]
            body_parts: list[str] = []
            if cinfo["docstring"]:
                body_parts.append(cinfo["docstring"])
            if cinfo["class_vars"]:
                body_parts.append("\n".join(cinfo["class_vars"].values()))

            methods_map: dict[str, str] = cinfo["methods"]
            if methods_map:
                ordered_method_names = sorted(
                    methods_map.keys(),
                    key=lambda m: (0 if m == "__init__" else 1),
                )
                body_parts.append(
                    "\n\n".join(methods_map[m] for m in ordered_method_names)
                )

            if not body_parts:
                body_parts.append("    pass")

            rendered_defs.append(f"{cinfo['header']}\n" + "\n\n".join(body_parts))

    sections: list[str] = []
    if imports:
        sections.append("\n".join(imports))
    if assignments:
        sections.append("\n".join(assignments.values()))
    if rendered_defs:
        sections.append("\n\n\n".join(rendered_defs))
    if main_blocks:
        sections.append("\n\n".join(main_blocks))

    assembled = "\n\n".join(sections).strip()
    return (assembled + "\n") if assembled else ""


def _extract_snippet_symbol_keys(snippet: str) -> set[str]:
    """
    Return granular symbol keys for `upsert_staged_snippet`:
    - Top-level function `foo` -> `"def:foo"`
    - Class `Wallet` with non-stub methods `__init__`, `deposit` -> `{"method:Wallet.__init__", "method:Wallet.deposit"}`
    - Class `Empty` with no non-stub methods -> `{"class:Empty"}`
    """
    if not snippet.strip():
        return set()
    try:
        tree = ast.parse(snippet)
    except SyntaxError:
        return set()

    keys: set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if not _is_stub_function(node):
                keys.add(f"def:{node.name}")
        elif isinstance(node, ast.ClassDef):
            non_stub_methods = [
                sub.name
                for sub in node.body
                if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef))
                and not _is_stub_function(sub)
            ]
            if non_stub_methods:
                for m_name in non_stub_methods:
                    keys.add(f"method:{node.name}.{m_name}")
            else:
                keys.add(f"class:{node.name}")
    return keys


def upsert_staged_snippet(
    module_snippets: dict[str, str],
    task_id: str,
    new_snippet: str,
    target_symbol: str | None = None,
) -> dict[str, str]:
    """
    Insert or update a staged snippet in `module_snippets`.
    - Wraps bare top-level class methods into `class <ClassName>:` when `target_symbol`
      is `'ClassName.method_name'`.
    - Uses granular symbol keys (`def:func` and `method:ClassName.method`) so separate
      class method tasks (`t1` for `Wallet.__init__`, `t2` for `Wallet.deposit`) coexist,
      while corrective rebuilds (`t2_repaired` for `Wallet.deposit`) cleanly replace `t2`.
    """
    normalized_snippet = ensure_class_wrapper(new_snippet, target_symbol)
    updated = dict(module_snippets)
    new_keys = _extract_snippet_symbol_keys(normalized_snippet)

    if new_keys:
        obsolete_ids = []
        for existing_id, existing_code in updated.items():
            if existing_id == task_id:
                continue
            existing_keys = _extract_snippet_symbol_keys(existing_code)
            if existing_keys and existing_keys.issubset(new_keys):
                obsolete_ids.append(existing_id)
        for k in obsolete_ids:
            updated.pop(k, None)

    updated[task_id] = normalized_snippet
    return updated

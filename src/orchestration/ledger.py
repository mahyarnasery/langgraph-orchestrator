from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Iterable

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
    # Collapse multi-line tracebacks into a single concise line to prevent prompt bloat
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
    for e in entries:
        lines.append(
            f"  {e.step}. [{e.status.upper()}] {e.layer}/{e.action} -> {e.target}: {e.detail}"
        )
    return "\n".join(lines)


def extract_python_signatures(source: str) -> list[str]:
    """Extract compact function and class signatures from a Python source string."""
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
            methods = [
                n.name
                for n in node.body
                if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
            ]
            sigs.append(f"class {node.name}[methods: {', '.join(methods)}]")
    return sigs


def _strip_self_imports(source: str, target_filename: str) -> str:
    """
    Remove accidental self-imports (e.g. `from analytics import parse_scores`
    inside `analytics.py`) that a Worker may write when seeing sibling functions.
    """
    module_stem = Path(target_filename.removeprefix("sandbox/")).stem
    cleaned_lines: list[str] = []
    for line in source.splitlines():
        stripped = line.strip()
        if re.match(rf"^from\s+{re.escape(module_stem)}\s+import\b", stripped):
            continue
        if re.match(rf"^import\s+{re.escape(module_stem)}\b", stripped):
            continue
        line = re.sub(rf"\b{re.escape(module_stem)}\.", "", line)
        cleaned_lines.append(line)
    return "\n".join(cleaned_lines)


def merge_staged_snippets_ast(
    target_filename: str,
    base_source: str,
    ordered_snippets: list[tuple[str, str]],
) -> str:
    """
    Deterministically assemble multiple staged Worker function/class snippets
    into a single coherent Python module using AST parsing and deduplication.
    """
    imports: list[str] = []
    seen_imports: set[str] = set()

    assignments: dict[str, str] = {}
    definitions: dict[str, str] = {}
    main_blocks: list[str] = []

    sources_to_merge: list[str] = []
    if base_source and base_source.strip():
        sources_to_merge.append(base_source)
    for _, snippet in ordered_snippets:
        if snippet and snippet.strip():
            sources_to_merge.append(_strip_self_imports(snippet, target_filename))

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

            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                definitions[node.name] = segment

            elif isinstance(node, (ast.Assign, ast.AnnAssign)):
                key = ast.unparse(node).split("=")[0].strip()
                assignments[key] = segment

            else:
                if segment not in main_blocks:
                    main_blocks.append(segment)

    sections: list[str] = []
    if imports:
        sections.append("\n".join(imports))
    if assignments:
        sections.append("\n".join(assignments.values()))
    if definitions:
        sections.append("\n\n\n".join(definitions.values()))
    if main_blocks:
        sections.append("\n\n".join(main_blocks))

    assembled = "\n\n".join(sections).strip()
    return (assembled + "\n") if assembled else ""

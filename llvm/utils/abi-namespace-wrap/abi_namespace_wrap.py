#!/usr/bin/env python3
# ===- abi_namespace_wrap.py - Migrate namespace llvm to macro ----*- py -*-===
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# ===-----------------------------------------------------------------------===
"""Rewrites top-level ``namespace llvm { ... }`` and
``namespace mlir { ... }`` blocks to use LLVM_NAMESPACE_BEGIN /
LLVM_NAMESPACE_END (and the MLIR analogues), per the ABI-namespace
migration plan.

This tool is the Stage 3a rewriter. The plan calls for a LibTooling-based
tool; this Python implementation is used for the tree-wide bulk
application because the scope (~5800 files) makes iterating on a
LibTooling build prohibitive. The Stage 5 "rewriter lifecycle decision"
chooses whether to promote this to a LibTooling tool or retire it.

Scope rules (see migration plan §2):

  * Only rewrites at file scope, depth 0.
  * Skips any ``namespace llvm`` or ``namespace mlir`` opening that is
    lexically inside an ``extern "C"`` block.
  * Skips files whose absolute path matches any of the ``--exclude-path``
    prefixes (canonical C-API list is the plan's §2 exclusion list).
  * Idempotent: a second run over the same files produces no changes.

Usage::

    abi_namespace_wrap.py --scope-root llvm/include/llvm/Support \\
                          --exclude-path llvm/include/llvm-c/ \\
                          --exclude-path clang/include/clang-c/ \\
                          --exclude-path mlir/include/mlir-c/ \\
                          --exclude-path mlir/lib/CAPI/ \\
                          --audit-out audit.json \\
                          [FILE ...]

If no FILEs are given, all files under the --scope-root directories are
scanned. The default file globs are `*.h`, `*.hpp`, `*.cpp`, `*.cc`,
`*.c` (the .c is necessary because a handful of LLVM .c files include
C++-guarded headers).
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import re
import sys
from pathlib import Path
from typing import Iterable, List, Optional, Tuple


# --- Tokenizer ----------------------------------------------------------

@dataclasses.dataclass
class _ScannerState:
    text: str
    i: int = 0
    n: int = 0

    def __post_init__(self):
        self.n = len(self.text)


def _skip_line_comment(s: _ScannerState) -> None:
    # s.i points at the first '/' of '//'
    s.i += 2
    while s.i < s.n and s.text[s.i] != "\n":
        s.i += 1


def _skip_block_comment(s: _ScannerState) -> None:
    s.i += 2
    while s.i + 1 < s.n:
        if s.text[s.i] == "*" and s.text[s.i + 1] == "/":
            s.i += 2
            return
        s.i += 1
    s.i = s.n  # unterminated; consume rest


def _skip_string(s: _ScannerState) -> None:
    s.i += 1  # opening "
    while s.i < s.n:
        c = s.text[s.i]
        if c == "\\" and s.i + 1 < s.n:
            s.i += 2
            continue
        if c == '"':
            s.i += 1
            return
        s.i += 1


def _skip_char(s: _ScannerState) -> None:
    s.i += 1  # opening '
    while s.i < s.n:
        c = s.text[s.i]
        if c == "\\" and s.i + 1 < s.n:
            s.i += 2
            continue
        if c == "'":
            s.i += 1
            return
        s.i += 1


def _skip_raw_string(s: _ScannerState) -> None:
    # s.i is at 'R' of R"delim(...)delim"
    # Consume R"
    s.i += 2
    delim_start = s.i
    while s.i < s.n and s.text[s.i] != "(":
        s.i += 1
    delim = s.text[delim_start:s.i]
    terminator = ")" + delim + '"'
    s.i += 1  # skip (
    idx = s.text.find(terminator, s.i)
    if idx < 0:
        s.i = s.n
        return
    s.i = idx + len(terminator)


def _is_ident_start(c: str) -> bool:
    return c.isalpha() or c == "_"


def _is_ident_cont(c: str) -> bool:
    return c.isalnum() or c == "_"


def _at_raw_string(s: _ScannerState) -> bool:
    # R"  or  uR"  or  UR"  or  u8R"  or  LR"
    i = s.i
    if i >= s.n:
        return False
    c = s.text[i]
    # Skip optional encoding prefix
    j = i
    for prefix in ("u8R", "uR", "UR", "LR", "R"):
        if s.text.startswith(prefix, j):
            j += len(prefix)
            break
    else:
        return False
    return j < s.n and s.text[j] == '"' and j == i + (len(prefix) - 1)


# Recognizers for `namespace X {` starting at a specific index.
_NS_RE = re.compile(
    r"namespace[\s\\\n]+"
    r"(?P<name>[A-Za-z_][A-Za-z_0-9:]*(?:\s*::\s*[A-Za-z_][A-Za-z_0-9]*)*)"
    r"[\s\\\n]*\{"
)


# --- Core rewrite logic -------------------------------------------------

@dataclasses.dataclass
class Edit:
    file: str
    kind: str           # wrap_decl, wrap_cxx17_nested
    open_start: int     # byte offset of `namespace` keyword in original text
    open_end: int       # byte offset just past the `{`
    close_start: int    # byte offset of matching `}`
    close_end: int      # byte offset just past the `}`
    ns_first: str       # `llvm` or `mlir`
    nested_rest: str    # for C++17 `namespace llvm::foo::bar {` this is
                        # `foo::bar`; empty for simple `namespace llvm {`


def _find_namespace_openings(text: str) -> List[Edit]:
    """Walk the file, find every top-level (depth 0, not under
    ``extern "C"``) opening of ``namespace llvm {`` or ``namespace mlir {``,
    plus their matching closing braces."""
    edits: List[Edit] = []
    s = _ScannerState(text=text)

    depth = 0
    # Stack of (depth, "extern_c" | "namespace_match" | "other"). We only
    # care about extern_c for suppression and namespace_match for finding
    # the close.
    stack: List[Tuple[int, str, Optional[int]]] = []
    # extern "C" { depth = depth at which we entered the block
    # If inside any extern "C" block, skip nested namespace rewrites.
    extern_c_depths: set = set()

    # Map open-depth for a pending edit to its index in `edits`.
    pending: List[int] = []  # stack of edit indices awaiting close

    while s.i < s.n:
        c = s.text[s.i]
        # Raw string?
        if _at_raw_string(s):
            # Advance past encoding prefix + R"
            # _skip_raw_string expects s.i at R
            # Find where R is
            j = s.i
            for prefix in ("u8R", "uR", "UR", "LR", "R"):
                if s.text.startswith(prefix, j):
                    j += len(prefix) - 1  # leave `s.i` at 'R'
                    break
            s.i = j
            _skip_raw_string(s)
            continue

        if c == "/" and s.i + 1 < s.n:
            if s.text[s.i + 1] == "/":
                _skip_line_comment(s)
                continue
            if s.text[s.i + 1] == "*":
                _skip_block_comment(s)
                continue

        if c == '"':
            _skip_string(s)
            continue
        if c == "'":
            _skip_char(s)
            continue

        # extern "C" detection
        if c == "e" and s.text.startswith("extern", s.i):
            end_kw = s.i + len("extern")
            if end_kw < s.n and not _is_ident_cont(s.text[end_kw]):
                # Look for optional whitespace then "C" or "C++"
                j = end_kw
                while j < s.n and s.text[j] in " \t\n\r":
                    j += 1
                if j < s.n and s.text[j] == '"':
                    k = j + 1
                    while k < s.n and s.text[k] != '"':
                        k += 1
                    lang = s.text[j + 1:k]
                    if lang == "C":
                        # Skip to the `{` or `;` after the string.
                        m = k + 1
                        while m < s.n and s.text[m] in " \t\n\r":
                            m += 1
                        if m < s.n and s.text[m] == "{":
                            # Entering extern "C" block at depth -> depth+1
                            extern_c_depths.add(depth + 1)
                            s.i = m + 1
                            depth += 1
                            continue
                        # Otherwise `extern "C" decl;` — no block; treat as
                        # a normal statement.
                        s.i = k + 1
                        continue

        # namespace <name> { at depth 0 (and not inside extern "C")
        if (
            depth == 0
            and not (extern_c_depths & {d for d in range(1, depth + 1)})
            and c == "n"
            and s.text.startswith("namespace", s.i)
        ):
            end_kw = s.i + len("namespace")
            if end_kw < s.n and not _is_ident_cont(s.text[end_kw]):
                m = _NS_RE.match(s.text, s.i)
                if m:
                    full_name = m.group("name")
                    first = full_name.split("::", 1)[0]
                    rest = full_name.split("::", 1)[1] if "::" in full_name else ""
                    # Canonicalize rest: strip whitespace around `::`
                    rest = re.sub(r"\s*::\s*", "::", rest)
                    if first in ("llvm", "mlir"):
                        open_start = s.i
                        open_end = m.end()
                        edits.append(Edit(
                            file="", kind="",
                            open_start=open_start, open_end=open_end,
                            close_start=-1, close_end=-1,
                            ns_first=first, nested_rest=rest,
                        ))
                        pending.append(len(edits) - 1)
                        depth += 1  # we just passed the {
                        s.i = open_end
                        continue

        # Any { / }
        if c == "{":
            depth += 1
            s.i += 1
            continue
        if c == "}":
            depth -= 1
            if depth < 0:
                # Unbalanced file; abort edits.
                return []
            if extern_c_depths and (depth + 1) in extern_c_depths:
                extern_c_depths.discard(depth + 1)
            # Close a pending namespace edit?
            if pending:
                # The top pending's depth at opening was (depth+1-1)
                # We only match close when depth returns to the
                # namespace's original outer depth (i.e. depth == 0
                # after we decrement, since openings are at file
                # scope).
                if depth == 0:
                    top_idx = pending.pop()
                    edits[top_idx].close_start = s.i
                    edits[top_idx].close_end = s.i + 1
            s.i += 1
            continue

        s.i += 1

    # Discard edits that weren't matched (open without close).
    edits = [e for e in edits if e.close_start >= 0]

    # Assign kind
    for e in edits:
        if e.nested_rest:
            e.kind = "wrap_cxx17_nested"
        else:
            e.kind = "wrap_decl"
    return edits


_LLVM_INCLUDE = '#include "llvm/Support/Compiler.h"'
_MLIR_INCLUDE = '#include "mlir/Support/ABINamespace.h"'
# Headers that already transitively provide the macros. Compiler.h is the
# canonical LLVM transport; ABINamespace.h is the direct form (rare).
_LLVM_INCLUDE_EXISTING = (
    "llvm/Support/Compiler.h",
    "llvm/Support/ABINamespace.h",
    "llvm/Support/ABINamespaceTag.h",
)
_MLIR_INCLUDE_EXISTING = (
    "mlir/Support/ABINamespace.h",
    "mlir/Support/ABINamespaceTag.h",
)


def _needs_include(text: str, kind: str) -> bool:
    """``kind`` is ``'llvm'`` or ``'mlir'``. Return True if the file does
    not already include a header that transports the required macros."""
    existing = _LLVM_INCLUDE_EXISTING if kind == "llvm" else _MLIR_INCLUDE_EXISTING
    for h in existing:
        # Match both #include "..." and #include <...>
        if f'"{h}"' in text or f"<{h}>" in text:
            return False
    return True


def _insert_include(text: str, include_line: str) -> str:
    """Insert ``include_line`` at a sensible spot.

    Strategy:
      * If the file already has any ``#include`` line, insert immediately
        after the last ``#include`` that appears before the first
        ``namespace`` / ``LLVM_NAMESPACE_BEGIN`` / ``MLIR_NAMESPACE_BEGIN``
        keyword.
      * Else, insert right after the include guard ``#define FOO_H`` line
        if one is present.
      * Else, insert at the top of the file after any license-block
        comments.

    Keeps a single trailing newline."""
    # Find a cutoff: stop searching after first namespace/macro keyword.
    m_ns = re.search(
        r"(?m)^(?:namespace\s+(?:llvm|mlir)|LLVM_NAMESPACE_BEGIN|MLIR_NAMESPACE_BEGIN)",
        text,
    )
    cutoff = m_ns.start() if m_ns else len(text)

    inc_matches = list(
        re.finditer(r'(?m)^#\s*include\s+[<"][^>"]+[>"][^\n]*\n', text[:cutoff])
    )
    if inc_matches:
        last = inc_matches[-1]
        insert_at = last.end()
        return text[:insert_at] + include_line + "\n" + text[insert_at:]

    m_def = re.search(r"(?m)^#\s*define\s+\w+(?:_H|_HPP|_INC)_?\s*\n", text[:cutoff])
    if m_def:
        insert_at = m_def.end()
        # Add a blank line between the guard and our new include for
        # readability.
        return text[:insert_at] + "\n" + include_line + "\n" + text[insert_at:]

    # No include guard / no includes. Skip leading license-block (// or /* */)
    # at file start.
    i = 0
    n = len(text)
    while i < n:
        if text.startswith("//", i):
            j = text.find("\n", i)
            if j < 0:
                break
            i = j + 1
            continue
        if text.startswith("/*", i):
            j = text.find("*/", i + 2)
            if j < 0:
                break
            i = j + 2
            if i < n and text[i] == "\n":
                i += 1
            continue
        if text[i] in " \t\n":
            i += 1
            continue
        break
    return text[:i] + include_line + "\n" + text[i:]


def _apply_edits(text: str, file_path: str, edits: List[Edit]) -> Tuple[str, List[dict]]:
    """Given edits over ``text``, return the rewritten text and an audit
    list. Edits are applied in reverse order so earlier offsets stay
    valid."""
    audit: List[dict] = []
    if not edits:
        return text, audit

    sorted_edits = sorted(edits, key=lambda e: e.open_start, reverse=True)
    buf = text
    # Pre-compute line numbers.
    line_starts = [0]
    for i, ch in enumerate(text):
        if ch == "\n":
            line_starts.append(i + 1)

    def lineof(off: int) -> int:
        import bisect
        return bisect.bisect_right(line_starts, off)

    for e in sorted_edits:
        ns_kw = "LLVM" if e.ns_first == "llvm" else "MLIR"
        begin_token = f"{ns_kw}_NAMESPACE_BEGIN"
        end_token = f"{ns_kw}_NAMESPACE_END"

        if e.nested_rest:
            new_open = f"{begin_token}\nnamespace {e.nested_rest} {{"
            new_close = f"}}\n{end_token}"
        else:
            new_open = begin_token
            new_close = end_token

        # Preserve a trailing comment on the open line if any, e.g.
        # `namespace llvm { // some note` -> keep the `// some note` after
        # the replaced token. The safe approach: the open replacement
        # covers exactly `namespace NAME {`, NOT any trailing whitespace
        # or comment on the same line. Leave the remainder intact.
        open_repl_start = e.open_start
        open_repl_end = e.open_end

        # Preserve a trailing `} // namespace llvm` style comment by
        # ONLY replacing the `}` char.
        close_repl_start = e.close_start
        close_repl_end = e.close_end

        buf = (buf[:close_repl_start] + new_close + buf[close_repl_end:])
        buf = (buf[:open_repl_start] + new_open + buf[open_repl_end:])

        audit.append({
            "file": file_path,
            "kind": e.kind,
            "ns_first": e.ns_first,
            "nested_rest": e.nested_rest,
            "open_line": lineof(e.open_start),
            "close_line": lineof(e.close_start),
        })

    # Inject the transport include if needed. Decide from the set of
    # first-identifiers we rewrote.
    firsts = {e.ns_first for e in edits}
    if "llvm" in firsts and _needs_include(buf, "llvm"):
        buf = _insert_include(buf, _LLVM_INCLUDE)
    if "mlir" in firsts and _needs_include(buf, "mlir"):
        buf = _insert_include(buf, _MLIR_INCLUDE)

    return buf, audit


def _file_within(path: Path, roots: List[Path]) -> bool:
    try:
        rp = path.resolve()
    except (FileNotFoundError, OSError):
        return False
    for r in roots:
        try:
            rp.relative_to(r)
            return True
        except ValueError:
            continue
    return False


def _file_excluded(path: Path, excludes: List[str]) -> bool:
    path_str = str(path.resolve()) + "/"
    abs_path = str(path.resolve())
    for ex in excludes:
        if ex in abs_path:
            return True
    # Also apply the universal `(llvm|clang|mlir)-c/` pattern per plan §2.
    if re.search(r"(?:^|/)(?:llvm|clang|mlir)-c(?:$|/)", abs_path):
        return True
    # Any file that itself defines a `extern "C"` region covering the
    # whole file (via a top-level extern "C" { ... }) is handled by the
    # tokenizer's depth-skip logic, not by file exclusion.
    return False


def _iter_source_files(roots: List[Path]) -> Iterable[Path]:
    exts = {".h", ".hpp", ".cpp", ".cc", ".c", ".inc"}
    for root in roots:
        if not root.exists():
            continue
        if root.is_file():
            if root.suffix in exts:
                yield root
            continue
        for p in root.rglob("*"):
            if p.is_file() and p.suffix in exts:
                yield p


def _already_migrated(text: str) -> bool:
    """Cheap heuristic: if the file already contains our macro names,
    consider it migrated and skip."""
    return ("LLVM_NAMESPACE_BEGIN" in text) or ("MLIR_NAMESPACE_BEGIN" in text)


def _process_file(
    path: Path,
    *,
    dry_run: bool,
) -> Tuple[int, List[dict]]:
    """Returns (num_edits_applied, audit_entries)."""
    try:
        raw = path.read_bytes()
    except OSError:
        return 0, []
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        # Try latin-1 for legacy files.
        text = raw.decode("latin-1")

    if _already_migrated(text):
        return 0, []

    edits = _find_namespace_openings(text)
    if not edits:
        return 0, []

    new_text, audit = _apply_edits(text, str(path), edits)
    if new_text == text:
        return 0, []
    if not dry_run:
        path.write_text(new_text, encoding="utf-8")
    return len(edits), audit


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scope-root", action="append", default=[],
                   help="Directory to scan (repeatable).")
    p.add_argument("--exclude-path", action="append", default=[],
                   help="Substring of absolute path to exclude (repeatable).")
    p.add_argument("--audit-out", default=None,
                   help="JSON output path for the audit log.")
    p.add_argument("--dry-run", action="store_true",
                   help="Do not write files; only compute the diff.")
    p.add_argument("files", nargs="*",
                   help="Explicit file list. If omitted, walk --scope-root.")
    args = p.parse_args(argv)

    excludes = list(args.exclude_path)
    roots = [Path(r).resolve() for r in args.scope_root]

    if args.files:
        file_iter: Iterable[Path] = [Path(f).resolve() for f in args.files]
    else:
        file_iter = _iter_source_files(roots)

    total_files = 0
    touched_files = 0
    total_edits = 0
    audit: List[dict] = []

    for path in file_iter:
        # If we have scope roots, ensure path is within them.
        if roots and not _file_within(path, roots):
            continue
        if _file_excluded(path, excludes):
            continue
        total_files += 1
        n, entries = _process_file(path, dry_run=args.dry_run)
        if n > 0:
            touched_files += 1
            total_edits += n
            audit.extend(entries)

    summary = {
        "files_scanned": total_files,
        "files_touched": touched_files,
        "edits": total_edits,
        "dry_run": args.dry_run,
    }

    if args.audit_out:
        Path(args.audit_out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.audit_out, "w") as f:
            json.dump({"summary": summary, "edits": audit}, f, indent=2)

    print(json.dumps(summary), file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())

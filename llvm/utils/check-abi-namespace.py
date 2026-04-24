#!/usr/bin/env python3
"""Check that LLVM/MLIR C++ source files use the ABI namespace macros.

This script is intended to run in CI after a change lands.  It looks for:

  * Raw ``namespace llvm {`` or ``namespace mlir {`` at file scope that
    should have been wrapped in ``LLVM_NAMESPACE_BEGIN`` /
    ``MLIR_NAMESPACE_BEGIN``.

  * Raw ``} // namespace llvm`` / ``} // namespace mlir`` closings that
    match such openings.

  * Out-of-line ``struct llvm::Foo { ... }`` / ``class mlir::Bar { ... }``
    definitions that should use ``LLVM_ABI_NS::`` / ``MLIR_ABI_NS::``.

Files under the excluded directory list (tests, C-API headers, third-party
sources) are skipped because they are outside the tagged scope.

Exit status:
  0  no violations
  1  at least one violation found
  2  invalid invocation

Usage:
    llvm/utils/check-abi-namespace.py [--root PATH] [--verbose]

The script is intentionally lightweight — a grep-level backstop rather
than a full C++ parser.  It is a complement, not a replacement, for the
rewriter (``llvm/utils/abi-namespace-wrap/abi_namespace_wrap.py``).
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path
from typing import Iterable, List, Tuple


# Directories (relative to --root) that must NOT be scanned: test fixtures,
# C API translation units, vendored third-party sources, and the rewriter
# tool itself (which has intentional raw strings for test input).
EXCLUDE_DIRS: Tuple[str, ...] = (
    # Synthetic test inputs — compiled out of the build system's reach.
    "clang/test",
    "clang-tools-extra/test",
    "lld/test",
    "lldb/test",
    "llvm/test",
    "mlir/test",
    "flang/test",
    "bolt/test",
    "polly/test",
    # C API headers — intentionally live in plain `namespace llvm`/`mlir`.
    "llvm/include/llvm-c",
    "mlir/include/mlir-c",
    # LLVMDemangle — intentionally tag-free so external symbolicators
    # (perf, gdb, llvm-objdump) see stable cross-version symbols, and so
    # it can be vendored without dragging in LLVMSupport.  See the
    # "Demangle library" section in docs/ABIInlineNamespace.rst.
    "llvm/include/llvm/Demangle",
    "llvm/lib/Demangle",
    # compiler-rt is a runtime that does not link against LLVMSupport.
    # Its handful of files in `namespace llvm { ... }` are either shared
    # on-disk data formats (MemProfData.inc, CtxInstrContextNode.h) that
    # must stay ABI-stable against untagged tools, or sanitizer test
    # helpers that live in-tree but build without LLVM includes.
    "compiler-rt",
    # Vendored third-party code that LLVM does not own.
    "third-party",
    "llvm/utils/gn",
    "llvm/utils/release",
    "llvm/utils/vim",
    "llvm/utils/emacs",
    "llvm/utils/Target",
    # The rewriter's own fixtures — they include intentional raw-string
    # samples of the old format.
    "llvm/utils/abi-namespace-wrap/test",
    "clang-tools-extra/clang-llvm-namespace-wrap/test",
    # Build artifacts (harmless if absent).
    "build",
)

# Source extensions that participate in the C++ ABI.
CXX_EXTS: Tuple[str, ...] = (
    ".h", ".hpp", ".hh", ".hxx",
    ".cpp", ".cc", ".cxx",
    ".inc",
)


# Regexes for the three violation classes.
NAMESPACE_OPEN_RE = re.compile(r"^\s*namespace\s+(llvm|mlir)\s*\{")
NAMESPACE_CLOSE_RE = re.compile(r"^\s*\}\s*//\s*(end\s+)?namespace\s+(llvm|mlir)\b")
OUT_OF_LINE_DEF_RE = re.compile(r"^\s*(struct|class|template\s+class)\s+(llvm|mlir)::")


def _strip_noncode(line: str) -> str:
    """Return ``line`` with string literals and `//` / `/* */` comments removed.

    This is a conservative, single-line approximation — good enough for a
    line-level brace counter.  Block comments opened on a different line
    are not tracked (they're rare in practice and the remaining false
    positives are harmless).
    """
    out: List[str] = []
    i = 0
    while i < len(line):
        c = line[i]
        # `//` line comment — drop the rest of the line.
        if c == "/" and i + 1 < len(line) and line[i + 1] == "/":
            break
        # `/* */` block comment — skip up to the closer on the same line.
        if c == "/" and i + 1 < len(line) and line[i + 1] == "*":
            end = line.find("*/", i + 2)
            if end < 0:
                break
            i = end + 2
            continue
        # String literal — find the matching closer, skipping escapes.
        if c == '"':
            j = i + 1
            while j < len(line) and line[j] != '"':
                if line[j] == "\\" and j + 1 < len(line):
                    j += 2
                    continue
                j += 1
            i = j + 1
            continue
        # Character literal.
        if c == "'":
            j = i + 1
            while j < len(line) and line[j] != "'":
                if line[j] == "\\" and j + 1 < len(line):
                    j += 2
                    continue
                j += 1
            i = j + 1
            continue
        out.append(c)
        i += 1
    return "".join(out)


def is_excluded(rel: str) -> bool:
    """True if a path (relative to the tree root) is in an excluded area."""
    # Normalise to forward-slash form so the comparison works on Windows too.
    rel_fwd = rel.replace(os.sep, "/")
    for prefix in EXCLUDE_DIRS:
        if rel_fwd == prefix or rel_fwd.startswith(prefix + "/"):
            return True
    return False


def iter_sources(root: Path) -> Iterable[Path]:
    """Yield every C++ source file under root that is not excluded."""
    root = root.resolve()
    for dirpath, dirnames, filenames in os.walk(root):
        # Prune excluded directories in-place so os.walk does not descend.
        pruned: List[str] = []
        for d in dirnames:
            rel = str(Path(dirpath, d).relative_to(root))
            if is_excluded(rel) or d.startswith("."):
                pruned.append(d)
        for d in pruned:
            dirnames.remove(d)

        for name in filenames:
            if not name.endswith(CXX_EXTS):
                continue
            path = Path(dirpath, name)
            rel = str(path.relative_to(root))
            if is_excluded(rel):
                continue
            yield path


def scan_file(path: Path) -> List[Tuple[int, str, str]]:
    """Return a list of (line_no, kind, line_text) violations in the file."""
    violations: List[Tuple[int, str, str]] = []
    # Track whether the current line is the continuation of a `#define`
    # directive body. Patterns inside a macro body are expected to be raw
    # (the rewriter deliberately does not transform them) and the matching
    # body-side `LLVM_NAMESPACE_BEGIN` version already lives inside
    # llvm/Support/ABINamespace.h itself.
    #
    # Also maintain a running brace depth so that nested `namespace llvm {`
    # blocks (e.g. a locally-scoped helper namespace inside another
    # project's namespace) are not misreported.  Only top-level (depth 0)
    # raw openings are true violations.
    try:
        with path.open("r", encoding="utf-8", errors="replace") as f:
            in_define = False
            depth = 0
            for i, line in enumerate(f, start=1):
                stripped = line.rstrip("\n")

                # Skip obvious comment lines — they are not code and do
                # not contribute to the brace depth.
                lstripped = stripped.lstrip()
                if lstripped.startswith("//") or lstripped.startswith("*"):
                    in_define = in_define and stripped.endswith("\\")
                    continue

                # Enter/continue a `#define` body on lines that start with
                # `#define` (possibly after leading whitespace and comments).
                if lstripped.startswith("#define") or in_define:
                    in_define = stripped.endswith("\\")
                    continue

                # Evaluate violation status BEFORE updating brace depth
                # because the regex matches the start of a line that itself
                # opens a brace.  The check is valid only when the
                # *previous* depth (depth before this line's braces) is 0.
                only_at_top = (depth == 0)

                if only_at_top and NAMESPACE_OPEN_RE.match(stripped):
                    violations.append((i, "raw-namespace-open", stripped))
                if only_at_top and NAMESPACE_CLOSE_RE.match(stripped):
                    # Report only if the matching opener was at top level
                    # (i.e. depth will go negative, which we clamp below).
                    violations.append((i, "raw-namespace-close", stripped))
                if only_at_top and OUT_OF_LINE_DEF_RE.match(stripped):
                    violations.append((i, "raw-out-of-line", stripped))

                code = _strip_noncode(stripped)
                opens = code.count("{")
                closes = code.count("}")
                depth += opens - closes
                if depth < 0:
                    depth = 0
    except OSError as e:
        print(f"warning: could not read {path}: {e}", file=sys.stderr)
    return violations


def main(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(
        description="Check that LLVM/MLIR C++ sources use ABI namespace macros.",
    )
    parser.add_argument(
        "--root",
        default=".",
        help="Root of the monorepo to scan (default: cwd).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print a summary line per scanned file.",
    )
    args = parser.parse_args(argv)

    root = Path(args.root).resolve()
    if not root.is_dir():
        print(f"error: --root {root!r} is not a directory", file=sys.stderr)
        return 2

    total_files = 0
    total_violations = 0
    violating_files: List[Tuple[Path, List[Tuple[int, str, str]]]] = []

    for path in iter_sources(root):
        total_files += 1
        viols = scan_file(path)
        if viols:
            total_violations += len(viols)
            violating_files.append((path, viols))
        if args.verbose:
            marker = "!" if viols else "."
            print(f"{marker} {path.relative_to(root)} ({len(viols)})")

    if violating_files:
        print(
            f"\nABI-namespace check: {total_violations} violations in "
            f"{len(violating_files)} file(s) (scanned {total_files})."
        )
        print()
        for path, viols in violating_files:
            rel = path.relative_to(root)
            for line_no, kind, text in viols:
                print(f"{rel}:{line_no}: {kind}: {text.strip()}")
        print()
        print("Suggested fixes:")
        print("  * raw-namespace-open  -> LLVM_NAMESPACE_BEGIN / MLIR_NAMESPACE_BEGIN")
        print("  * raw-namespace-close -> LLVM_NAMESPACE_END / MLIR_NAMESPACE_END")
        print("  * raw-out-of-line     -> LLVM_ABI_NS:: / MLIR_ABI_NS::")
        print()
        print("See llvm/docs/ABIInlineNamespace.rst for authoring guidance.")
        return 1

    print(
        f"ABI-namespace check: clean ({total_files} files scanned, "
        f"no raw `namespace llvm`/`namespace mlir` violations)."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

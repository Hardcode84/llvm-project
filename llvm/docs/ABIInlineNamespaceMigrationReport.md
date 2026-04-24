# ABI Inline Namespace Migration — Postmortem Report

Branch: `llvm-inline-namespace`
Base: `main` (merge-base `3cd4a795d9fd`)
Tip: `4b2e36511f44`
Commits: 27
Overall diff: **6 363 files changed, +22 576 / −13 570** (net +9 006)

> This report covers a tree-wide refactor that wraps every
> `namespace llvm { ... }` and `namespace mlir { ... }` block in an inline
> versioned namespace (e.g. `namespace llvm { inline namespace v23_0 { ... } }`)
> so two independently-built copies of the LLVM/MLIR C++ API can coexist in
> one process without C++ symbol collisions.  The main reference doc is
> [`ABIInlineNamespace.rst`](ABIInlineNamespace.rst).

---

## 1. Executive summary

The migration **delivered the core technical mechanism** (distinct Itanium
mangling for the tagged C++ surface) and has the full tree building and
passing its test suites with the default `v${MAJOR}_${MINOR}` tag active.
Every major subproject (LLVM, MLIR, Clang, LLD, Flang, Polly, BOLT, LLDB)
compiles and runs under the new macro wrapping.

However, **the gap between the initial goal and the final state is
larger than the commit count suggests**:

- A tail of documented **untagged "ABI-stable zones"** — Demangle,
  ProfileData + coverage, compiler-rt headers, the C API, and a long
  tail of consumer files that have to match those zones' raw namespace —
  **still collides** when two LLVMs are co-loaded.  Exact file count
  depends on where you draw the boundary (non-test C++ under the
  documented zones: a few hundred files; including `compiler-rt` and
  shared headers: several thousand).
- The tag alone does not enable dual-LLVM coexistence: the distributor
  must *also* ship with distinct SONAME, `-Wl,-Bsymbolic`, and private
  RUNPATH, plus static-link any untagged zones they care about.  These
  are now documented but **not enforced by CMake**.
- The CI backstop (`llvm/utils/check-abi-namespace.py`) exists and
  reports clean, but **is not wired into upstream CI**.
- The rewriter took **four bugfix commits** after the first bulk pass
  and required one **tree-wide rescan**; no clang-based tooling was
  used.

Feasibility verdict: **partly achieved**.  A distributor can ship two
co-loaded LLVMs today if they follow the documented steps and accept
the untagged-zone caveats.  For the average LLVM consumer, the change
is source-compatible and invisible.

---

## 2. Goal-vs-reality breakdown

| # | Stated goal | Delivered | Status |
|---|---|---|---|
| G1 | Tag every `namespace llvm` / `namespace mlir` C++ symbol | Tagged the overwhelming majority of non-test C++ in the main subprojects; documented carveouts (Demangle, ProfileData + profile/coverage consumers, C API, `compiler-rt`, plus operational exclusions for `*/test/`, `third-party/`, editor/GN utils) are kept raw | **Partial** |
| G2 | Two LLVMs with distinct tags coexist in one process, no C++ collisions (tagged zone) | Mangling is distinct; unit test proves cross-TU identity in a single build but **does not prove dual-DSO coexistence** | **Partial (mechanism present, not end-to-end tested)** |
| G3 | C API remains source/ABI-compatible | `extern "C"` surface untouched; headers under `llvm-c/`, `mlir-c/` never wrapped | **Yes** |
| G4 | C API provides runtime isolation of two LLVMs | Not possible by design: `extern "C"` symbols are process-globally shared; documented as a fundamental limitation | **No (scope correction)** |
| G5 | Default build is tagged | `LLVM_ABI_NAMESPACE` defaults to `v${MAJOR}_${MINOR}`; empty string disables | **Yes** |
| G6 | Downstream CMake sees the tag | `LLVMConfig.cmake` exports it as read-only variable | **Yes** |
| G7 | Downstream CMake *enforces* matching tag on consumers | `LLVMConfig.cmake` does not compare consumer flags; doc warns "MUST NOT redefine" | **No (discipline-only)** |
| G8 | CI catches accidental raw `namespace llvm` | Backstop script exists and reports clean | **Partial (not wired into upstream CI)** |
| G9 | ProfileData is tagged | Reverted to raw; shares byte-identical `.inc` with compiler-rt that cannot include `LLVMSupport` | **No (scope correction)** |
| G10 | Demangle is tagged | Reverted to raw; needed for external symbolicators (`perf`, `llvm-objdump`) | **No (scope correction)** |
| G11 | compiler-rt is tagged | Never tagged; its runtime libs don't link `LLVMSupport` | **No (scope correction)** |
| G12 | Global state (cl::opt, PassRegistry) is per-tag | Tagged registries use tagged symbols and are per-DSO; ProfileData `cl::opt` stays shared | **Partial** |
| G13 | TableGen emitters emit the macros | `NamespaceEmitter` updated; all `.td` tests refreshed | **Yes** |
| G14 | clang-format handles the new macros | `.clang-format` `StatementMacros` updated; Polly switched to `-style=file` | **Yes** |

Net: **7 green, 5 partial, 4 explicit scope corrections**.  Every
scope correction is documented in the RST; none were found after
merge.

---

## 3. Commit taxonomy and effort breakdown

### 3.1 Buckets

Each commit was placed in exactly one bucket:

All line numbers below are sums of `git show --shortstat` per commit;
the per-bucket percentages use the **whole-branch diff churn** (22 576
+ 13 570 = 36 146 insertions + deletions) as denominator.  Rounding
means the column does not sum exactly to 100 %.

| Bucket | Commits | Lines (+ / −) | % of total churn |
|---|---:|---:|---:|
| infra (macros, CMake, initial scripts) | 2 | +1 103 / −6 | 3.1 % |
| rewriter-bugfix | 4 | +167 / −41 | 0.6 % |
| bulk-rewrite (Stage 3b B0..B8, rescan, Stage 5d) | 11 | +19 665 / −13 531 | 91.8 % |
| cleanup (TableGen, out-of-line, lit tests, format) | 6 | +190 / −122 | 0.9 % |
| policy (default flip, carveouts, CI backstop) | 2 | +845 / −173 | 2.8 % |
| docs (plan + final narrowing) | 2 | +1 062 / −153 | 3.4 % |

(Per-commit shortstats sum to slightly more than the whole-branch diff
because the same line is sometimes rewritten by a later commit.  The
rightmost column uses the whole-branch figure, which is why
`91.8 + 3.1 + 0.6 + 0.9 + 2.8 + 3.4 ≈ 102.6 %`.)

Take-aways:

- **~92 %** of the line-churn is mechanical wrapping.  The hand-written
  cleanup work accounts for less than 1 % of lines but fixed
  non-mechanical mangling issues the plan did not anticipate.
- **Rewriter bugfixes cost 208 lines** but sit upstream of ~33 k lines
  of bulk churn — roughly a 160× leverage factor.
- **Policy commits are small** (2 commits, 1 018 lines) but contain the
  highest-value content: default-flip, carveout definition, and the CI
  backstop.

### 3.2 Bulk-rewrite breakdown (per Stage 3b subtree)

| Stage | Commit | Files | Lines | Subtree |
|---|---|---:|---:|---|
| B0 | `3ad84953c9be` | 3 | 631 | Rewriter + TableGen emitter |
| B1 | `0b08fdb89f6d` | 274 | 1 298 | `llvm/{include,lib}/Support` |
| B2 | `a530eb1a6788` | 111 | 526 | `llvm/include/llvm/ADT` |
| B3 | `14ef3df37ec2` | 1 069 | 5 178 | `llvm/{include,lib}/{IR,Analysis,Transforms,CodeGen}` |
| **B3.5** | `9d8bac3cf181` | **2 229** | **11 196** | Remaining `llvm/{include,lib}/*` |
| B4 | `ea832db1756e` | 492 | 2 639 | `llvm/{tools,unittests,utils}` |
| B5 | `b2c2dcb77cec` | 1 240 | 7 395 | `mlir/{include/mlir,lib}` (minus C API) |
| B6 | `b12c1c8cec84` | 179 | 952 | `mlir/{tools,unittests,test,examples,benchmark,utils}` |
| B7 | `929aa4201bb3` | 214 | 1 180 | `clang/{include,lib,tools,unittests,utils,examples}` |
| B8 | `dfe8d2ddf8a1` | 162 | 912 | `lld/`, `flang/`, `clang-tools-extra/` |
| Rescan | `2a8188fee229` | 22 | 100 | Missed files after rewriter fixes |
| Stage 5d | `1a13ba3d4b9c` | 350 | 1 820 | BOLT, LLDB, Polly, offload, libc |

Top-3 subtrees by churn: **B3.5 (remaining LLVM core) > B5 (MLIR core)
> B3 (LLVM IR/Analysis/Transforms/CodeGen)**.  Together they sum to
23 769 lines, which is **~72 %** of the bulk-rewrite churn and
**~66 %** of the whole branch.

B3.5 was not in the original plan — it was added mid-stream when B1–B3
turned out to miss `llvm/lib/{Target, MC, Object, DebugInfo, ...}`.

### 3.3 Rework signals

- **Multi-touch rate:** 87 files were edited by ≥ 2 commits
  (1.4 % of 6 363).  4 were edited by ≥ 3 commits.  The only file
  edited by 5 commits is the rewriter itself.
- **Multi-touched file hotspots:** `llvm/lib/ProfileData/InstrProfReader.cpp`
  (3 edits — bulk → carveout), the LLVM and MLIR ABI namespace headers
  (3 edits each — design iteration), most `llvm/tools/llvm-profgen/*`
  files (2 edits — wrapped then un-wrapped for the ProfileData
  carveout).
- **Rewrite-cycle commits:** 4 rewriter bugfixes plus 1 rescan commit
  (5 by role).  Three of these five have `fix` / `handle` / `Rescan`
  literally in the subject line.
- **Loops in the timeline:**
  1. After B0, an immediate small loop (`#define`-body skip) before B1.
  2. After B8, **three** rewriter bugfix commits landed (raw-string
     detection, preprocessor brace imbalance, digit separator), and
     later — after the out-of-line / TableGen cleanup commits — a
     rescan re-wrapped 22 missed files.
  That's two discovery → fix → apply cycles overall; the second cycle
  was interrupted by the `LLVM_ABI_NS` and TableGen-emitter work and
  resumed with the rescan.

### 3.4 Top-10 most-changed files

| Lines | File |
|---:|---|
| 745 | `llvm/utils/abi-namespace-wrap/abi_namespace_wrap.py` |
| 614 | `llvm/docs/ABIInlineNamespace.rst` |
| 564 | `inline-namespace-migration-plan.md` |
| 332 | `llvm/utils/check-abi-namespace.py` |
| 109 | `llvm/unittests/Support/YAMLIOTest.cpp` |
| 109 | `llvm/lib/Target/AArch64/Utils/AArch64BaseInfo.cpp` |
| 108 | `llvm/unittests/Support/ABINamespaceTest.cpp` |
| 84 | `llvm/include/llvm/Support/ABINamespace.h` |
| 77 | `clang/lib/APINotes/APINotesYAMLCompiler.cpp` |
| 55 | `mlir/include/mlir/Support/ABINamespace.h` |

Infrastructure (scripts, the macro header, tests) dominates the top-10;
the rest of the churn is spread thinly across ~6 000 files at roughly
6–7 lines each (1 `_BEGIN` / 1 `_END` / 1 `#include Compiler.h` + occasional
out-of-line fixups).

---

## 4. Unexpected issues

Issues the original plan did not anticipate, in rough order of cost.

### 4.1 Itanium out-of-line mangling (high cost)

The plan assumed wrapping `namespace llvm { ... }` in an inline namespace
was sufficient.  It wasn't: **Itanium C++ ABI mangles out-of-line
definitions by the literal qualifier**, so

```cpp
void llvm::Foo::bar() {}     // mangled as llvm::Foo::bar — wrong!
```

continues to mangle into the outer namespace even when `llvm::Foo` is
defined inside `llvm::v23_0`.  Result: tagged builds link-failed on
`undefined symbol: llvm::v23_0::Foo::bar`.

Fix: introduce `LLVM_ABI_NS` and `MLIR_ABI_NS` macros (`llvm::vX_Y` /
`mlir::vX_Y` when active) and require every out-of-line definition to
use them:

```cpp
void LLVM_ABI_NS::Foo::bar() {}          // mangles into the tagged ns
template class LLVM_ABI_NS::SmallVectorBase<int>;
```

Hit by: every out-of-line DenseMapInfo specialization, DOTGraphTraits
specialization, explicit template instantiation, and a long tail of
friend declarations.  Took three follow-up commits (`f1c43c66e826`,
`f5a690435ada`, plus `8338beb22583` for TableGen emitters) to land.

### 4.2 TableGen emission (medium cost)

TableGen emitters produced raw `namespace llvm { ... }` strings that
leaked into dozens of `.inc` outputs downstream.  Fix: teach
`llvm::NamespaceEmitter` (`llvm/include/llvm/TableGen/CodeGenHelpers.h`)
to emit the macros when the target namespace starts with `llvm` or
`mlir` (commit `8338beb22583`, `+55 / −14`).  13 LLVM lit-test
expected-output files and 3 MLIR lit-test files had to be regenerated
(commits `bef0e91211eb` and `7ecfc83c1624`).

### 4.3 ProfileData / compiler-rt shared `.inc` files (high cost — caused scope reduction)

`llvm/include/llvm/ProfileData/MemProfData.inc` and friends are
**byte-identical** with files under `compiler-rt/include/profile/`
(verified by `diff`).  compiler-rt does not link `LLVMSupport` and
cannot pick up `LLVM_NAMESPACE_BEGIN`.  After the initial bulk wrap:

- The tagged-LLVM side expected `llvm::v23_0::MemProfRecord`.
- compiler-rt built its own copy that defined `llvm::MemProfRecord`
  (raw).
- C++ link errors on anything that crossed the boundary — profgen
  reading compiler-rt output, Clang's coverage instrumentation, etc.

**Resolution:** revert ProfileData to raw `namespace llvm` tree-wide —
including `llvm-profgen`, `llvm-cov/CoverageFilters.h`,
`Support/Discriminator.h`, and consumer headers in `Analysis/` and
`Transforms/` that forward-declare ProfileData types or hold
`extern cl::opt` declarations for options defined in untagged
ProfileData.

This is a **real reduction of the migration's scope** and is the
largest single unexpected issue.  Expanding the carveout rippled into
~10 files of Clang CodeGen and Instrumentation `.cpp`s.  The bulk of
the structural work landed in `b59b1facdcaf`
(Stage 5a/5b/5d — untagged ABI-stability zones + docs + backstop),
with the final doc/backstop alignment in `4b2e36511f44`.

### 4.4 Demangle library (medium cost — caused scope reduction)

`LLVMDemangle` is used by external symbolicators and debuggers that
expect stable cross-version symbol names like `llvm::itaniumDemangle`.
It is also designed standalone (no `LLVMSupport` dep).  Tagging it
created tagged/untagged conflicts against consumers that still expected
the raw names.

**Resolution:** keep `llvm/include/llvm/Demangle` and
`llvm/lib/Demangle` raw, use the pre-existing
`DEMANGLE_NAMESPACE_BEGIN` macro in `DemangleConfig.h` (which was
already the Demangle-internal namespace control), document the zone as
untagged.

### 4.5 `extern cl::opt` cross-boundary declarations (medium cost)

Consumer files in `lib/Transforms/Instrumentation/` declared `extern
cl::opt<bool> EnableVTableProfileUse` inside `LLVM_NAMESPACE_BEGIN` but
the definition lived in untagged ProfileData.  Linker errors ensued.

**Resolution:** document the "`extern` must match the namespace of its
definition" pattern, add an authoring section, add affected `.cpp`
files to the backstop exclusion list.

### 4.6 Nested `namespace coverage {` / `namespace sampleprof {` (medium cost)

Files that opened `namespace llvm { namespace coverage { ... } }` but
were consumed through `using namespace coverage;` created ambiguity
when only the outer was tagged.  Fixed by keeping the inner
`coverage` / `sampleprof` blocks consistently outside the tagged region
(matching ProfileData).  Affected `clang/lib/CodeGen/CoverageMappingGen.*`
and `llvm/tools/llvm-profgen/*.h`.

### 4.7 Rewriter lexer completeness (low cost but annoying)

The rewriter is a hand-rolled character scanner (not clang-based).
Four edge cases were missed on the first bulk pass:

- `namespace` inside `#define` bodies (wrapped as if file-scope).
- Raw string literals containing a `{ namespace llvm ...` pattern.
- Preprocessor conditionals with unbalanced braces across branches.
- C++14 digit separators (`1'000'000`) confused the string/char-literal
  skipper into thinking a `}` was still inside a string.

Each fix was one small rewriter commit + re-run on affected files.
Together they cost ~200 lines and one tree-wide rescan.

### 4.8 Polly's `check-format` target (low cost)

Polly's CMake invoked `clang-format -style=llvm` directly, ignoring the
repo's `.clang-format`.  The new namespace macros were indented as
function-call statements.  Fix: switch to `-style=file` and add the
macros to `StatementMacros` in the repo `.clang-format`.

---

## 5. Pain points

In priority order by total time-cost:

### P1. Regex-based rewriter has a lexer-completeness tax

The rewriter is 745 lines of Python doing a hand-rolled character scan.
Every time a new C++ lexical construct trips it up (raw strings, digit
separators, `#define` bodies), the fix is localised but the **audit
cost is tree-wide**: you can never fully trust that the previous bulk
pass saw every file correctly, so every rewriter change motivates a
rescan.  A clang-based tool (libtooling or clang-refactor) would
inherit the standard's lexer for free and eliminate this class of bug
entirely, at the cost of a heavier build dependency and slower
iteration.

Section 6 below lists five remaining risks in the rewriter and three
known blind spots in the backstop.  None are blocking; all are
symptomatic of the "second lexer" maintenance burden.

### P2. Out-of-line definitions were not caught by the rewriter

The rewriter wraps `namespace llvm { ... }` blocks.  It does **not**
rewrite out-of-class definitions like `void llvm::Foo::bar() {}` at
file scope — those need `LLVM_ABI_NS::` qualification and had to be
fixed by a separate pass.  This fix-up is what the cleanup bucket
mostly contains.

### P3. Two scripts, two lexers, imperfect parity

The rewriter and the CI backstop each implement their own approximate
C++ lexer.  They drift: the backstop scans `.hh`/`.hxx`/`.cxx` while
the rewriter does not; the backstop does not recognise C++17 nested
`namespace llvm::foo {` form that the rewriter targets; the rewriter
processes `mlir/test/` while the backstop excludes it.  None of these
gaps broke the migration, but every one of them is a latent "quiet
regression" path.

### P4. ABI-stable untagged zones were not in the plan

The original plan (`inline-namespace-migration-plan.md`, committed in
`a7da26efeb3d`) named only the **C API** as an explicit untagged zone.
During the bulk pass we discovered that Demangle, ProfileData, coverage
format headers, and a long tail of consumer files also could not be
tagged — either because they share byte-identical headers with
`compiler-rt` or because external tools depend on their stable symbols.
Every one of these carveouts is a post-hoc scope reduction relative to
the plan, not an anticipated boundary, and the ProfileData carveout
required a second structural refactor (tagged forward declarations of
LLVM core types from within untagged ProfileData bodies) to keep the
build compiling.

### P5. Downstream tag enforcement is advisory only

`LLVMConfig.cmake` exports `LLVM_ABI_NAMESPACE` to downstream CMake
consumers but does not **compare** it to anything the consumer sets.
The doc warns "do not redefine `LLVM_ABI_NAMESPACE` on your compile
line; it must match the installed header."  A misconfigured consumer
silently produces an ODR-violating build that the linker cannot
diagnose — a quiet, high-impact footgun.

### P6. Unit tests prove cross-TU identity, not cross-DSO coexistence

`llvm/unittests/Support/ABINamespaceTest.cpp` proves that an object
built in TU1 is usable from TU2 through the inline namespace, and that
`typeid` reflects the tag when active.  It does **not** build two
DSOs with distinct tags and dlopen them into one process.  The
mechanism is correct by construction (distinct mangled names →
distinct symbols), but there is no end-to-end test that catches
loader-level regressions.

### P7. Upstream CI does not run the backstop

`check-abi-namespace.py` is checked into `llvm/utils/` and reports
clean, but upstream CI does not invoke it.  A future contributor can
land a raw `namespace llvm {` and the tree will compile fine — the
tag just won't apply to their code.  Catching the regression depends
entirely on distributor-side wiring.

---

## 6. Migration scripts — deep dive

### 6.1 Rewriter (`llvm/utils/abi-namespace-wrap/abi_namespace_wrap.py`, 745 lines)

Hand-rolled character scanner plus a regex (`_NS_RE`) that matches
`namespace` followed by an arbitrary `::`-separated identifier chain
and an opening brace.  The `llvm` / `mlir` check is performed in
Python by splitting the captured name at `::` and testing the first
segment.  C++17 nested-namespace form (`namespace llvm::foo::bar {`)
is therefore supported.  Attribute-qualified forms like
`namespace [[deprecated]] llvm {` are not matched.

The scanner maintains brace depth, skips line comments, block
comments, strings, character literals (with C++14 digit separator
handling for `'` after a digit/identifier), and raw string literals.
Preprocessor lines are consumed as a logical block so `namespace`
tokens inside `#define` bodies are ignored.  `extern "C"` blocks bump
depth so openings inside them are not file-scope (`extern "C++"` is
not special-cased).  `#if`/`#else`/`#elif` branches get depth
snapshotted/restored to survive preprocessor-split braces.  Edits
replace the outer `namespace llvm { ... }` pair with
`LLVM_NAMESPACE_BEGIN` / `LLVM_NAMESPACE_END` (or the MLIR pair) and
insert an `#include "llvm/Support/Compiler.h"` or
`#include "mlir/Support/ABINamespace.h"` if the file doesn't already
include an equivalent header.

**Evolution (5 commits):**

| Commit | Fix | Bug class |
|---|---|---|
| `3ad84953c9be` | Initial implementation | — |
| `08f580070a5e` | Skip `namespace` inside `#define` bodies | Preprocessor interaction |
| `34a8a4e07f64` | Fix raw string literal detection | Lexer completeness |
| `a230e7a6d006` | Handle `#if`/`#else` brace imbalance | Preprocessor interaction |
| `abc9a1b23bc8` | Treat `'` in numeric literals as digit separator | Lexer completeness (C++14) |

**Risks still present:**

- Idempotency check (`_already_migrated`) is a substring match on
  `LLVM_NAMESPACE_BEGIN` or `MLIR_NAMESPACE_BEGIN`; any occurrence
  anywhere in a file skips the whole file, even if other regions
  legitimately need rewriting.
- Exclusion check (`_file_excluded`) is also substring-based; a short
  `--exclude-path` argument can over-match.
- Attribute-qualified namespaces (`namespace [[deprecated]] llvm {`)
  would not match the regex.
- Trigraph/digraph input is not handled.
- `extern "C++"` blocks are not special-cased.

### 6.2 CI backstop (`llvm/utils/check-abi-namespace.py`, 332 lines)

Walks the tree, filters by extension (`.h`/`.hpp`/`.hh`/`.hxx`/`.cpp`/
`.cc`/`.cxx`/`.inc`), prunes paths in `EXCLUDE_DIRS`, and line-scans
for three patterns:

- `namespace (llvm|mlir) {` at the start of a line with running
  brace-depth 0 → `raw-namespace-open` violation.
- `} // namespace (llvm|mlir)` matching close → `raw-namespace-close`
  violation.
- `(struct|class|template class) (llvm|mlir)::` at file scope →
  `raw-out-of-line` violation (catches the out-of-line class definition
  case that the rewriter does not fix).

Depth tracking does **not** preprocess the source — it counts literal
`{` / `}` in a strip-commented-and-stringed version of each line.
`LLVM_NAMESPACE_BEGIN` appears alone on a line without a literal brace
and therefore contributes 0 to the running depth, which is the
intended behaviour for a top-level-only scanner but also means the
backstop cannot detect a file that correctly opens with the macro and
then illegally re-opens raw `namespace llvm {` inside a tagged
block.

**Known blind spots:**

- C++17 nested `namespace llvm::foo {` form is not matched — the
  rewriter handles it but the backstop does not flag a regression to
  it.
- `_strip_noncode` does not recognise C++14 digit separators; brace
  counts inside numeric literals with `'` can drift.
- Block comments `/* ... */` spanning multiple lines are only
  approximated via the leading-`*` heuristic.

### 6.3 Exclusion list vs. the documented "ABI-stable untagged zones"

The backstop's `EXCLUDE_DIRS` now aligns with the doc's Section
"ABI-stable untagged zones" — enforced by running the scanner and
iterating until the tree is clean.  Categories:

| Category | Files | Purpose |
|---|---|---|
| Test trees | `*/test/` in every subproject | Tests contain intentional raw namespaces (negative cases, fixtures) |
| C API | `llvm/include/llvm-c`, `mlir/include/mlir-c` | `extern "C"` — not eligible for tagging |
| Demangle | `llvm/{include,lib}/Demangle` | Standalone lib, external-tool ABI |
| compiler-rt | `compiler-rt/` | Standalone runtimes, no `LLVMSupport` dep |
| ProfileData | `llvm/{include,lib}/ProfileData`, `llvm/tools/llvm-profgen` | Byte-identical `.inc` with compiler-rt |
| ProfileData consumers | `CoverageFilters.h`, `Discriminator.h`, `MemoryProfileInfo.h`, `StaticDataProfileInfo.{h,cpp}`, `MemProfUse.h`, `ProfiledCallGraph.h`, `InstrProfiling.cpp`, `PGOInstrumentation.cpp`, `IndirectCallPromotion.cpp` | Forward-declare or `extern` untagged types |
| ProfileData unit tests | `MemProfTest.cpp`, `DataAccessProfTest.cpp`, `MemProfUseTest.cpp` | Must match untagged type definitions |
| Clang CodeGen/coverage | `clang/lib/CodeGen/CoverageMappingGen.{h,cpp}`, `CodeGenModule.h` | Bridges to untagged `llvm::coverage` |
| Third-party / editor / GN / release / build | `third-party`, `llvm/utils/{gn,release,vim,emacs,Target}`, `build` | Not authored by LLVM, not C++, or intermediate build output |
| Rewriter fixtures | `llvm/utils/abi-namespace-wrap/test`, `clang-tools-extra/clang-llvm-namespace-wrap/test` | Intentionally raw input for rewriter tests; the `clang-tools-extra` entry is a forward-looking listing — the directory does not exist in upstream today |

Current backstop output: `ABI-namespace check: clean
(37 887 files scanned, no raw 'namespace llvm'/'namespace mlir' violations).`

---

## 7. Feasibility verdict

**Partly achieved.**

A distributor who follows the documented recipe — vendor-specific tag,
distinct SONAME, `-Wl,-Bsymbolic`, private RUNPATH, static-linked
untagged zones — can ship a compiler that embeds LLVM and coexists in
the same process as the host's LLVM *for the tagged C++ surface*.  The
mangling isolation is real and covers the overwhelming majority of
non-test C++ in the main subprojects.

The documented untagged zones — C API, Demangle, ProfileData (plus
its `llvm-profgen`, `llvm-cov` and CodeGen coverage consumers), and
the `compiler-rt` tree — are **permanently shared** between the two
LLVMs.  For most use cases (a compiler plugin that doesn't touch
profile data or expose LLVMContext through its own public API) this
is not a problem.  For a plugin that, say, reads a `memprof` profile
emitted by the host's compiler-rt, the distributor must either
static-link ProfileData, confine profile work to one LLVM, or vendor
the relevant types.

The migration did **not** deliver a self-contained "turn on the flag
and be done" mode for distributors.  It delivered a *toolbox*
(mangling, tag export, backstop, doc) that a competent distributor
can assemble into a working dual-LLVM distribution.  Whether that
counts as "success" depends on whether you were expecting a turnkey
feature or infrastructure.

---

## 8. Recommendations for a similar migration from scratch

In priority order.

### R1. Start with a clang-based rewriter

Spend the first week building a `clang-refactor` or libtooling-based
tool instead of hand-rolling a Python scanner.  You pay ~1 week of
setup cost and inherit the standard's lexer forever.  The four
rewriter bugfix commits in this migration (raw strings, digit
separators, `#define` bodies, preprocessor conditionals) would not
have existed.  More importantly, you can rewrite **out-of-line
definitions** in the same pass, eliminating the entire cleanup
bucket.

Trade-off: iteration is slower (build LLVM first), and you need a
real test harness of `.cpp` fixtures.  Worth it.

### R2. Do the ABI-stable zone audit *before* the bulk pass

Spend a few days cataloguing:

- Every `.inc` / header shared with `compiler-rt` or other standalone
  runtimes.
- Every library that external tools (`perf`, `libc++abi`,
  symbolicators) expect to find with stable symbols.
- Every `extern cl::opt` whose definition lives in a library that
  can't adopt the macros.

This is the list that becomes your untagged-zone registry.  Having it
up-front means the bulk rewrite can skip them from day one.  In this
migration the audit happened post-hoc: `b59b1facdcaf` bundled the
structural refactor (~74 files) and `4b2e36511f44` landed the
doc/backstop alignment.  A prior audit would have avoided the
wrap-then-revert cycles for Demangle and ProfileData trees.

### R3. Write a dual-DSO unit test

Build two tiny shared libraries, each tagged differently, dlopen both
into a test binary, and assert that `typeid(A::Foo) != typeid(B::Foo)`
and that each library's globals are independent.  This is the test
the unit-test suite currently *doesn't* have, and it's the only test
that actually exercises the migration's stated goal.  The existing
`ABINamespaceTest.cpp` + two companion TUs proves cross-TU identity
within one build; a dual-DSO test would need its own small CMake
fixture and a second tagged library.

### R4. Ship tag enforcement in CMake, not docs

Make `LLVMConfig.cmake` compare the consumer's
`-DLLVM_ABI_NAMESPACE=...` against the installed tag and error out on
mismatch.  Document the escape hatch (set it to empty to opt out) but
refuse silent ODR violations.  Right now the doc says "must match"
and the build system doesn't check.

### R5. Decide up front whether "dual-LLVM coexistence" is the goal

The plan said "two LLVMs in one process"; the final state says
"*some* of two LLVMs in one process, with caveats and a recipe."
Both are legitimate outcomes, but they want different engineering
investments:

- **Turnkey coexistence** requires tagging Demangle (with a separate
  shim for external tools), tagging ProfileData (with a new shared
  format that both LLVM and compiler-rt can read without includes),
  and either moving the C API behind a tagged dispatch layer or
  accepting it as shared.  That is a substantial follow-on project of
  its own — on the order of each of the four zones being roughly the
  same scope as the ProfileData carveout effort.
- **Toolbox coexistence** is what this migration delivered: mangling
  + docs + backstop, distributor wires the rest.  That's the 27
  commits you see.

Pick one at the planning stage.  The cost of the toolbox approach is
that every distributor reinvents the same SONAME/RUNPATH/static-link
incantation; the cost of the turnkey approach is fighting
`compiler-rt` and `libc++abi` compatibility.

### R6. Wire the CI backstop into upstream CI on day one

`check-abi-namespace.py` is ~330 lines of pure-stdlib Python and
catches every regression the migration cares about.  Add it as a
premerge check before the bulk pass even starts, so the Stage 3b
commits are validated in CI instead of by hand.  The rescan commit
(`2a8188fee229`) would have been caught on the first bulk commit
instead of after B8.

### R7. Budget for the preprocessor tax

A significant share of the "unexpected" cost (points 4.3–4.5) is
downstream consequences of the C preprocessor's incompatibility with
the migration: headers shared across projects that can't pick up the
macros, `extern` declarations that must match their definitions'
namespace spelling, and forward declarations that span the tagged
boundary.  Plan for it: a two-day design exercise on "how does a
tagged header forward-declare an untagged type and vice versa" would
have saved a week of iterative debugging.

### R8. Don't skip the `LLVM_ABI_NS` macro story

The Itanium out-of-line mangling issue is the single most confusing
thing about this migration for a new contributor and it is easy to
miss.  Lead the design doc with it, not with "this is basically just
`inline namespace`."

---

## 9. What worked well

- **Staged bulk rollout (B1 → B8).**  Splitting the bulk rewrite into
  subtree-sized commits meant each landed as a manageable code review
  and any regression was localised.  Mean bulk-rewrite commit size
  (12 commits including B0, rescan, and Stage 5d): ~530 files, ~2 800
  insertions + deletions — large but reviewable.
- **Default-off → bulk-wrap → default-on.**  Wrapping everything with
  `LLVM_ABI_NAMESPACE=""` first meant the bulk commits were
  mechanical and the whole tree kept passing tests before the tag was
  turned on.  The default-flip (`635834200beb`) is a single `+7 / −4`
  commit on `llvm/CMakeLists.txt`.
- **Macros hide the tag from source code.**  No consumer code changed.
  `llvm::Value` still means `llvm::Value`.  This is the core virtue
  of `inline namespace` and it justified the whole approach.
- **The rewriter was cheap to iterate on.**  Each bugfix was a few
  dozen lines and a re-run of the rewriter over affected subtrees.
  The four rewriter-bugfix commits together account for 208 lines.
- **TableGen emitter update was clean.**  `NamespaceEmitter` owns
  the raw `namespace llvm {` strings; teaching it about the macros
  was a single `+55 / −14` commit (`8338beb22583`).
- **Documentation caught up.**  The final `ABIInlineNamespace.rst`
  documents the full untagged-zone set, the ELF distribution recipe,
  and the `extern cl::opt` cross-boundary pattern.  A distributor
  reading it today has enough to proceed.

---

## 10. Appendix: per-commit ledger

| Hash | Bucket | Files | Lines (+ / −) | Subject |
|---|---|---:|---:|---|
| `a7da26efeb3d` | docs | 1 | +564 / −0 | Add inline versioned namespace migration plan |
| `2bfd0c9b9010` | infra | 14 | +478 / −0 | Stages 1+2: inline versioned namespace infrastructure |
| `3ad84953c9be` | infra | 3 | +625 / −6 | B0 rewriter + TableGen emitter updates |
| `08f580070a5e` | rewriter-bugfix | 1 | +69 / −8 | Skip `namespace` inside `#define` bodies |
| `0b08fdb89f6d` | bulk-rewrite | 274 | +727 / −571 | B1: wrap `llvm/{include,lib}/Support` |
| `a530eb1a6788` | bulk-rewrite | 111 | +302 / −224 | B2: wrap `llvm/include/llvm/ADT` |
| `14ef3df37ec2` | bulk-rewrite | 1 069 | +2 954 / −2 224 | B3: wrap `llvm/{include,lib}/{IR,Analysis,Transforms,CodeGen}` |
| `9d8bac3cf181` | bulk-rewrite | 2 229 | +6 582 / −4 614 | B3.5: wrap remaining `llvm/{include,lib}/*` |
| `ea832db1756e` | bulk-rewrite | 492 | +1 587 / −1 052 | B4: wrap `llvm/{tools,unittests,utils}` |
| `b2c2dcb77cec` | bulk-rewrite | 1 240 | +4 525 / −2 870 | B5: wrap `mlir/{include/mlir,lib}` (minus C API) |
| `b12c1c8cec84` | bulk-rewrite | 179 | +582 / −370 | B6: wrap `mlir/{tools,unittests,test,examples,benchmark,utils}` |
| `929aa4201bb3` | bulk-rewrite | 214 | +693 / −487 | B7: wrap `clang/{include,lib,tools,unittests,utils,examples}` |
| `dfe8d2ddf8a1` | bulk-rewrite | 162 | +557 / −355 | B8: wrap `lld/`, `flang/`, `clang-tools-extra/` |
| `34a8a4e07f64` | rewriter-bugfix | 1 | +32 / −27 | Fix raw string literal detection |
| `a230e7a6d006` | rewriter-bugfix | 1 | +56 / −6 | Handle preprocessor conditional brace imbalance |
| `abc9a1b23bc8` | rewriter-bugfix | 1 | +10 / −0 | Treat `'` in numeric literals as digit separator |
| `f1c43c66e826` | cleanup | 3 | +26 / −2 | Add `LLVM_ABI_NS` / `MLIR_ABI_NS` for out-of-line definitions |
| `8338beb22583` | cleanup | 3 | +55 / −14 | Teach TableGen emitters to emit namespace macros |
| `f5a690435ada` | cleanup | 24 | +34 / −34 | Use `LLVM_ABI_NS` / `MLIR_ABI_NS` for out-of-line class/struct definitions |
| `7ecfc83c1624` | cleanup | 3 | +6 / −6 | Update mlir-tblgen test expectations for namespace macros |
| `2a8188fee229` | bulk-rewrite | 22 | +56 / −44 | Rescan after rewriter fixes: wrap missed files |
| `bef0e91211eb` | cleanup | 13 | +62 / −64 | Update TableGen lit tests for namespace macro emission |
| `635834200beb` | policy | 1 | +7 / −4 | Stage 4: Default `LLVM_ABI_NAMESPACE` to `v<MAJOR>_<MINOR>` |
| `1a13ba3d4b9c` | bulk-rewrite | 350 | +1 100 / −720 | Stage 5d: Apply rewriter to remaining subprojects |
| `b59b1facdcaf` | policy | 74 | +838 / −169 | Stage 5a/5b/5d: Untagged ABI-stability zones + docs + CI backstop |
| `d21aa9504376` | cleanup | 2 | +7 / −2 | Teach clang-format about namespace macros and use repo style in Polly |
| `4b2e36511f44` | docs | 4 | +498 / −153 | Doc: narrow scope claims, document untagged zones, align backstop |

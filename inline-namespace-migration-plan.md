# Inline Versioned Namespace Migration for LLVM/MLIR

Technical migration plan. Introduces inline versioned namespaces across `namespace llvm` and `namespace mlir` so that two independently-built copies of LLVM/MLIR can coexist in one process **for consumers that link the C++ API**. The C API (identical mangled names across copies by design) remains a single-copy surface in v1; multi-copy C API coexistence is a separate design problem (§12).

The canonical C API exclusion list is stated once in §2 and referenced by pointer elsewhere.

---

## 1. Problem

A single process can end up with two copies of LLVM/MLIR:

- Two components that each statically link LLVM, linked into one binary.
- A host binary that links LLVM plus a plugin (`.so` / `.dylib` / `.dll`) that also links LLVM.
- Multiple Python extensions (JAX, IREE, ONNX-MLIR, …) that each bundle LLVM and are loaded into one interpreter.

C++ symbols in the two copies have identical mangled names (`_ZN4llvm5Value...`). On ELF and Mach-O with default visibility the dynamic linker collapses them; consumers then call into the "wrong" copy and crash the first time they touch a per-copy static: `cl::opt` double-registration, `ManagedStatic` corruption, `TypeID` mismatch, RTTI assertion.

Existing partial mitigations each cover only part of the C++ surface:

| Mitigation | Covers | Does not cover |
|---|---|---|
| `-fvisibility=hidden` + `LLVM_ABI` annotations (`llvm/docs/InterfaceExportAnnotations.rst`) | Leaks from a dylib through unintended public symbols | Two copies statically linked into one binary; inlined header code |
| `-Bsymbolic-functions` | Intra-library call binding | Data symbols, RTTI, inter-library collisions; `-Bsymbolic` itself (but not `-Bsymbolic-functions`) breaks MLIR `TypeID` per `mlir/CMakeLists.txt:74-76` (matches `-Bsymbolic[^-]`, PR51420) |
| `-funique-internal-linkage-names` | TU-local statics after LTO | Anything at external linkage; anything in headers |
| `-Dllvm=llvmX` preprocessor substitution | Most external symbols | Non-namespaced statics, explicit instantiations, any site where `llvm::` is computed rather than literal |

Goal: rename every C++ symbol under `namespace llvm` / `namespace mlir` so the mangled name encodes a build-specific tag, while source-level spelling stays unchanged. An **inline versioned namespace** in the style of libstdc++'s `std::__cxx11` achieves this.

## 2. Goals and non-goals

### Goals

- Two independently-built LLVM/MLIR copies safely coexist in one process on ELF, Mach-O, and PE/COFF, for consumers that link the C++ API, as long as each consumer only talks to the copy it was built against.
- Silent wrong-copy crashes convert to link-time / `dlopen`-time failures when a consumer's headers and library disagree on the tag.
- Default end-user experience unchanged: a single-LLVM build with the tag knob at its default produces a bit-identical external-symbol set to today (exact property defined in §10).
- Downstream distributors get a supported knob to vendor-tag their LLVM via `-DLLVM_ABI_NAMESPACE=<value>`.

### Non-goals

- **ABI stability across tags.** The tag changes with the version on purpose — this plan separates copies, it does not make them interoperable. Every major-version bump of the default tag requires a full downstream rebuild; there is no mixed-tag binary compatibility (§3.3).
- **Touching the C API.** The rewriter, every build mode defined here, and every test must leave the C API untouched. The canonical exclusion list (referenced elsewhere as "the §2 exclusion list") is:
  - `llvm/include/llvm-c/` and any `.cpp` file that implements a `llvm-c/` declaration
  - `clang/include/clang-c/` and any `.cpp` file that implements a `clang-c/` declaration
  - `mlir/include/mlir-c/`
  - `mlir/lib/CAPI/`
  - any `extern "C"` region, no matter where it occurs
  - any file that matches the regex pattern `(llvm|clang|mlir)-c/`

  Multi-copy C API coexistence is a separate problem (ELF symbol versioning, versioned public names) and is out of scope (§12).
- **Fixing layout drift from mismatched headers.** Short accessors on `SmallVector`, `StringRef`, `ArrayRef`, etc. inline into the consumer's TU. Where consumer headers and the linked library disagree on struct layout, tagging cannot re-route already-inlined code. It does convert the failure mode from "silent crash" to "undefined symbol at link time" for every out-of-line call, which is where most of the value is.
- **Replacing `LLVM_ABI` annotations.** Orthogonal: `LLVM_ABI` controls *what* is exported from a dylib; the inline namespace controls *under what mangled name*. The two compose at every exported declaration (§7).

## 3. Design

### 3.1 Shape

Every top-level `namespace llvm { ... }` block is wrapped with an inline namespace whose name encodes the build's ABI tag:

```cpp
namespace llvm {
inline namespace LLVM_ABI_NAMESPACE {

// existing contents

}
}
```

`inline namespace` means:

- `llvm::Value` still names the same type at source level — forward declarations, friend decls, template arguments, explicit qualifications, specializations like `std::hash<llvm::StringRef>` keep working without change.
- The mangled name becomes `_ZN4llvm5v23_05ValueE` instead of `_ZN4llvm5ValueE`. (Itanium mangling: `<length><identifier>`. The tag `v23_0` is 5 characters, so the segment is `5v23_0`.)
- RTTI `type_info` for `llvm::Value` lives at a tag-dependent symbol, so `dynamic_cast` and `typeid` across two copies fail cleanly rather than silently succeed with the wrong vtable (this benefit applies only when RTTI is enabled for those types).

Nested namespaces (`llvm::sys`, `llvm::cl`, `llvm::detail`, …) inherit the tag for free. The same wrapping is applied to `namespace mlir`.

### 3.2 Macros

Two inputs drive the expansion, both generated by CMake (§3.3, Stage 2):

- `LLVM_ABI_NAMESPACE` — the tag identifier (e.g. `v23_0`). Set only when tagging is active.
- `LLVM_ABI_NAMESPACE_ACTIVE` — defined (as `1`) if and only if the tag is non-empty.

Gating on `LLVM_ABI_NAMESPACE_ACTIVE`, not on `#ifdef LLVM_ABI_NAMESPACE`, avoids the "defined-but-empty" footgun: `-DLLVM_ABI_NAMESPACE=` (empty replacement) would be defined for `#ifdef` and would expand `inline namespace {` — an anonymous inline namespace, which produces TU-local identities that silently break ODR across object files.

The macros live in a dedicated header `llvm/include/llvm/Support/ABINamespace.h` (included transparently from `Compiler.h` so existing headers pick them up without change):

```cpp
// llvm/include/llvm/Support/ABINamespace.h
#pragma once
#include "llvm/Support/ABINamespaceTag.h"  // generated by CMake; see §3.3

#ifdef LLVM_ABI_NAMESPACE_ACTIVE
#define LLVM_NAMESPACE_BEGIN namespace llvm { inline namespace LLVM_ABI_NAMESPACE {
#define LLVM_NAMESPACE_END   } }
#else
#define LLVM_NAMESPACE_BEGIN namespace llvm {
#define LLVM_NAMESPACE_END   }
#endif
```

Symmetric `mlir/include/mlir/Support/ABINamespace.h` for MLIR, guarded by `MLIR_ABI_NAMESPACE_ACTIVE`, emitting `MLIR_NAMESPACE_BEGIN` / `MLIR_NAMESPACE_END`. LLVM and MLIR keep separate macros (small duplication accepted; see the CI equivalence check in Stage 2).

Macro expansion invariants:

1. When tagging is inactive, expansion is plain `namespace llvm { ... }`. Mangled names are bit-identical to today, for every TU, in every compiler. There is no anonymous inline namespace anywhere in the expansion.
2. Consumer forward declarations must use the macros. A raw `namespace llvm { class Value; }` in a downstream header produces an untagged declaration that never resolves against `llvm::v23_0::Value` in the library. This is the largest migration task for downstreams; link failure is the signal.
3. The macros never appear inside `extern "C"` or in any file on the §2 exclusion list. Enforced by the rewriter (refuses to enter those regions and paths) and by the §10 symbol-diff guard post-build.

### 3.3 Tag composition

The tag is a single C++ identifier. CMake assembles it and validates it:

```cmake
set(LLVM_ABI_NAMESPACE "" CACHE STRING
    "Inline namespace tag for ABI isolation (e.g. v23_0). Empty = disabled.")

if(LLVM_ABI_NAMESPACE)
  if(NOT LLVM_ABI_NAMESPACE MATCHES "^[A-Za-z_][A-Za-z0-9_]*$")
    message(FATAL_ERROR
      "LLVM_ABI_NAMESPACE='${LLVM_ABI_NAMESPACE}' is not a valid C++ identifier. "
      "It must match ^[A-Za-z_][A-Za-z0-9_]*$.")
  endif()
  set(LLVM_ABI_NAMESPACE_ACTIVE 1)
endif()

configure_file(
  include/llvm/Support/ABINamespaceTag.h.in
  ${CMAKE_CURRENT_BINARY_DIR}/include/llvm/Support/ABINamespaceTag.h
  @ONLY)
```

`ABINamespaceTag.h.in` is the sole source of truth:

```cpp
// @generated from ABINamespaceTag.h.in
#pragma once
#cmakedefine LLVM_ABI_NAMESPACE @LLVM_ABI_NAMESPACE@
#cmakedefine LLVM_ABI_NAMESPACE_ACTIVE @LLVM_ABI_NAMESPACE_ACTIVE@
```

Consumers see the tag via `#include "llvm/Support/ABINamespace.h"` (direct) or any header that includes `Compiler.h` (transitive). Nothing is pushed via `-D` on the compile line; the tag exists in exactly two forms at build time — the CMake cache value and the generated header — and the header is the only form TUs ever observe.

Default: `LLVM_ABI_NAMESPACE = v${LLVM_VERSION_MAJOR}_${LLVM_VERSION_MINOR}` (e.g. `v23_0`) ships in Stage 4. Until then the CMake default is empty and tagging is opt-in.

Major-version upgrade contract: every release that changes the default tag requires a full rebuild of every downstream consumer. There is no mixed-tag interop window. Vendor suffixes (e.g. `v23_0_acme` for distributor `acme`) are recommended to prevent tag collisions across downstreams.

Throughout the doc, `vX_Y` is shorthand for whatever `v${LLVM_VERSION_MAJOR}_${LLVM_VERSION_MINOR}` resolves to; `v23_0` appears as a concrete example.

### 3.4 Rejected alternatives

- **ELF `.symver` / version scripts.** Works for dynamic dispatch across DSOs on ELF only. Does not help two static archives linked into one binary (identical unmangled linkage names before the linker applies version scripts), does not apply to PE/COFF, and does not isolate RTTI `type_info` or inline-defined symbols.
- **Compiler-level flag (`-fnamespace-version=` in Clang).** Would eliminate the tree-wide source rewrite. Requires Clang-only builds of LLVM (GCC-built LLVM would diverge), complicates the bootstrap story, needs its own compiler-feature work, and interacts badly with consumers that mix Clang and GCC.
- **Preprocessor substitution (`-Dllvm=llvmX`).** Breaks at every site where `llvm::` is computed rather than literal — friend declarations, template-template parameters, elaborated type specifiers, string macros. Partial coverage with silent gaps.
- **Narrow tagging (only known-colliding singletons).** Tag only `ManagedStatic<T>` storage, `cl::opt` registry, `PassRegistry`, `TypeID` roots. Smaller patch, but covers only the symbols currently known to collide. Does not isolate RTTI broadly, does not prevent silent vtable aliasing across arbitrary LLVM classes, does not prevent future additions from reintroducing collisions. The full-namespace approach covers everything by construction; a narrow approach trades completeness for patch size, and completeness is the whole point.

## 4. What this fixes

| Scenario | Before | After |
|---|---|---|
| Two copies with different tags statically linked into one binary | Linker collapses symbols, one copy "wins" silently | No collision; each copy operates on its own statics cleanly |
| Two copies with different tags, one as a `dlopen`'d plugin in a host that also links LLVM | Plugin calls interpose onto host's copy | Disjoint symbols, plugin stays with its own copy |
| Consumer built against tag A headers, loaded against tag B library | Silent crash when field offsets disagree | `ld` / `dlopen` fails with undefined `llvm::vA::Symbol` |
| Two copies both call `cl::ParseCommandLineOptions` | Abort on duplicate registration | Each copy has its own option registry |
| `ManagedStatic<T>` across copies | Racing mutation, invariant violations | Per-copy instance |

## 5. What this does not fix

- **Inlined code from mismatched headers.** If `SmallVector::size()` is defined in the header and the consumer saw layout A while the library was built with layout B, the inlined read happens at the wrong offset. The tag cannot re-route already-inlined code. Link-time failure catches the out-of-line case; inline mismatches remain undetectable at runtime in v1. Mitigation: consumers rebuild against the exact headers of the library they link; the build system enforces this via `LLVMConfig.cmake` (Stage 2).
- **Layout-affecting build-flag mismatches under the same tag.** Same version, different `LLVM_ENABLE_ABI_BREAKING_CHECKS` → same tag → silent breakage. Downstream distributors who need flag-awareness must fold the relevant flags into their `LLVM_ABI_NAMESPACE` override.
- **C API multi-copy coexistence.** Out of scope (§12).
- **Precompiled headers and C++20 module BMIs.** A BMI or PCH built with tag A used in a TU compiled with tag B is an ODR violation the linker cannot diagnose. The tag must be part of the cache key for any PCH/BMI caches downstream of LLVM headers. Build systems consuming LLVM headers must invalidate on tag change; this is documented in `LLVMConfig.cmake` (Stage 2).
- **`-fno-rtti` builds.** The RTTI-based isolation (§3.1) applies only when RTTI is enabled for the types in question. Tag-based symbol isolation still works; only the `dynamic_cast` / `typeid` failure mode is unavailable.
- **Runtime tag mismatch detection.** v1 relies on link-time failure exclusively. A runtime self-report API was considered and rejected for v1 because any such API is itself tagged — a mismatched consumer cannot even link to call it — and a stable `extern "C"` probe would add new C API surface that is out of scope. Revisit if downstream demand justifies it.

## 6. Worked example

Library build, LLVM 23.0 head:

```
-DLLVM_ABI_NAMESPACE=v23_0
-DLLVM_BUILD_LLVM_DYLIB=ON
-DLLVM_LINK_LLVM_DYLIB=ON
-DLLVM_ENABLE_LLVM_EXPORT_ANNOTATIONS=ON
-DCMAKE_CXX_VISIBILITY_PRESET=hidden
```

Produces `libLLVM-23.so` where `llvm::Value::Value()` mangles to `_ZN4llvm5v23_05ValueC2Ev`.

Downstream build against the same LLVM:

```cpp
#include "llvm/IR/Value.h"
llvm::Value *v = /* ... */;   // source compiles unchanged
v->getType();                  // mangles to _ZN4llvm5v23_05Value7getTypeEv
```

Resolves cleanly against the dylib.

Downstream build against a vendored fork using `-DLLVM_ABI_NAMESPACE=v23_0_acme` (10 chars → `10v23_0_acme`):

```cpp
#include "llvm/IR/Value.h"
llvm::Value *v = /* ... */;   // same source
v->getType();                  // mangles to _ZN4llvm10v23_0_acme5Value7getTypeEv
```

Both dylibs loaded in one process: zero collision.

### Minimum reproducer

From a clean `llvm-project/` checkout with tagging enabled:

```bash
cmake -S llvm -B ../llvm-build -DLLVM_ABI_NAMESPACE=v23_0 -DCMAKE_BUILD_TYPE=Release
ninja -C ../llvm-build LLVMSupport
nm --defined-only ../llvm-build/lib/libLLVMSupport.a | c++filt | grep -m3 'llvm::v23_0::'
```

The last command prints three mangled-then-demangled symbols confirming the tag is in the symbol table.

## 7. Relationship to existing ABI work

| Existing work | Interaction |
|---|---|
| `LLVM_ABI` annotations + hidden visibility (`llvm/docs/InterfaceExportAnnotations.rst`) | Complementary and composed. Every exported declaration uses both: outer namespace macros for naming, `LLVM_ABI` on the declaration for export. Stage 5 updates `InterfaceExportAnnotations.rst` with the combined pattern. |
| `-Bsymbolic` / `-Bsymbolic-functions` | Partially subsumed for C++. Versioned namespacing removes most of the motivation for `-Bsymbolic` because intra-library C++ references are no longer at risk of interposition. The `mlir/CMakeLists.txt:74-76` fatal-error guard matches `-Bsymbolic[^-]`, so `-Bsymbolic-functions` is allowed but bare `-Bsymbolic` is not; tagging does not automatically change this. Validation happens in Stage 3d. |
| `-funique-internal-linkage-names` | Complementary for internal-linkage statics, which are outside namespace scope. |
| `LLVM_BUILD_LLVM_DYLIB` on Windows | Unblocks one objection: DLLs need unique export names when two DLLs coexist; the tag makes that automatic. |
| MLIR `TypeID` (pointer-based identity from address-of-static) | Coexists. Tag-isolated MLIR produces distinct `TypeID` storage per copy — which is the intended behavior under the multi-copy model. Validated in Stage 3d. |
| C API ABI versioning | Orthogonal; separate problem (§12). |

## 8. Migration stages

### 8.0 Execution invariants

Four invariants hold across every stage below.

**(1) Bulk edits go through subagents.** Any change that mechanically transforms more than one file in a single pass (the Stage 3 rewriter runs, the Stage 5 tooling sweeps) is executed by a scoped subagent. The subagent prompt specifies input paths, tool invocation, build/test command, expected output (diff + structured audit log). The coordinator defines scopes, dispatches subagents, reviews their outputs, and merges PRs. The coordinator does not run bulk edits inline.

**(2) C API is off-limits.** Every rewriter and tooling invocation takes the §2 exclusion list as a startup argument (unknown paths are a fatal error to prevent typos silently widening scope). Every subagent prompt pointer-references the list. Every produced diff is gated on the §10 C API symbol-diff guard.

**(3) Build-directory layout.** Out-of-tree, sibling to `llvm-project/`:

| Directory | Purpose | Owner | Tag |
|---|---|---|---|
| `../llvm-build` | Main validation build | Coordinator | Current stage's tag (empty in Stages 1–2, `v23_0` in Stage 3 onward) |
| `../llvm-build-untagged` | Reference for the §10 C API symbol-diff guard | Coordinator | Always empty |
| `../llvm-build-<batch-id>` | Isolated per-batch verifier build | Subagent | Whatever the batch needs |

The coordinator reconfigures `../llvm-build` and `../llvm-build-untagged` at the start of each stage and between batch merges. Subagents create and tear down their per-batch dir, never touching `../llvm-build` or `../llvm-build-untagged`. All `ninja` invocations use `-C <dir>`; all `cmake` invocations use `-S llvm -B <dir>`.

**(4) Infrastructure requirements.** Eight parallel `../llvm-build-<batch-id>` dirs at ~40 GB each implies ~320 GB of disk and RAM sufficient for parallel `ninja -j` within each. Degraded mode (when infra is not available): run verifiers sequentially against one shared `../llvm-build-verify`, trading calendar time for disk. The plan assumes the full-parallel mode; the degraded mode is supported by the same commands with `N=1` verifier at a time.

### Stage 1 — Infrastructure

Land the macros. Tagging is opt-in at the CMake layer; default is off.

- Add `llvm/include/llvm/Support/ABINamespace.h` (hand-written) and `include/llvm/Support/ABINamespaceTag.h.in` (CMake template, output at configure time to the build tree).
- Have `llvm/include/llvm/Support/Compiler.h` include `ABINamespace.h` so every existing header transparently picks up the macros.
- Mirror for MLIR: `mlir/include/mlir/Support/ABINamespace.h` and `ABINamespaceTag.h.in`.
- Unit test (`llvm/unittests/Support/ABINamespaceTest.cpp`) linking two fixture TUs that define and use `llvm::TestType` via the macros. Verify via `nm` / demangle:
  - With tagging inactive, mangled name is `_ZN4llvm8TestType…` in both TUs; `typeid` equality across TUs holds; `dynamic_cast` works across TUs.
  - With `LLVM_ABI_NAMESPACE=v23_0`, mangled name is `_ZN4llvm5v23_08TestType…`; same type-identity and cross-TU properties, now under the tagged name.

**Exit criteria:** `ninja -C ../llvm-build check-llvm check-mlir` passes. `nm --defined-only ../llvm-build/lib/libLLVM*.so`, filtered to the set of exported symbols, is bit-identical before and after the stage (with tagging inactive in both configurations).

### Stage 2 — Build system integration

- Add `LLVM_ABI_NAMESPACE` as a CMake cache variable (default empty). Validate input: must match `^[A-Za-z_][A-Za-z0-9_]*$` or `fatal_error`. Set `LLVM_ABI_NAMESPACE_ACTIVE=1` iff non-empty.
- `configure_file` the two `ABINamespaceTag.h.in` templates into the build tree. Do not add `add_compile_definitions` — the generated header is the only transport.
- Export the effective tag value as a **read-only** variable (`LLVM_ABI_NAMESPACE`) in `LLVMConfig.cmake` and `MLIRConfig.cmake`. Contract documented in the same file: downstream consumers must not redefine the tag on their own compile line; they must pick it up via `find_package(LLVM)`. If a consumer sets a different tag, link will fail on the first cross-copy call — the §10 guard will not catch this because it operates in-tree.
- PCH / BMI cache guidance documented in `LLVMConfig.cmake`: consumers with PCH or C++20 module caches must include the tag value in the cache key.
- CI equivalence check: a small test asserts that `LLVM_NAMESPACE_BEGIN`/`_END` expansion matches `MLIR_NAMESPACE_BEGIN`/`_END` expansion under equivalent tag conditions (only outer identifier differs). Catches drift if one macro evolves without the other.

**Exit criteria:** `cmake -S llvm -B ../llvm-build -DLLVM_ABI_NAMESPACE=v23_0` configures cleanly; `ninja -C ../llvm-build check-llvm check-mlir` passes; `nm ../llvm-build/lib/libLLVM*.so | c++filt` shows tagged symbols. Invalid tag values are rejected at configure time. `../llvm-build-untagged` continues to produce bit-identical exported symbols to trunk.

### Stage 3 — Mechanical source transformation

#### 3a. Build the rewriter (coordinator, single session)

A LibTooling tool `clang-tools-extra/clang-llvm-namespace-wrap/` that:

- Matches top-level `namespace llvm { ... }` and `namespace mlir { ... }` declarations at file scope (including the C++17 nested form `namespace llvm::foo { ... }`, rewritten to the two-level form before wrapping).
- Rewrites to `LLVM_NAMESPACE_BEGIN ... LLVM_NAMESPACE_END`.

Hard invariants (each enforced by a separate lit test under the tool):

- Refuses to enter any `extern "C"` region.
- Refuses to modify any path on the §2 exclusion list. Unknown paths passed to `--exclude-path` are fatal.
- Handles forward declarations, template forward declarations, `std::hash` specializations, explicit instantiations, friend declarations, and out-of-class member definitions — each with a dedicated fixture and a dedicated test under §10's Design-rule tests, not "spot-check."
- Emits `audit.json` listing every location it changed: file, line range, kind (`wrap_decl`, `wrap_forward_decl`, `wrap_instantiation`, `wrap_specialization`, `wrap_friend`, `wrap_out_of_class_def`).

The §3c audit review categories and the `audit.json` `kind` values share the same vocabulary so that automation can map each `kind` to its review gate without ad-hoc translation.

The AST-matcher snippet in §9.1 is illustrative: the production matcher set is richer than the snippet suggests (explicit handling of out-of-class definitions, friend decls, `namespace std` specializations, included `.inc` files — see below).

**TableGen and `.inc` files.** Generated `.inc` files (TableGen emitters in `llvm/utils/TableGen/`, MLIR table-gen, etc.) contain `namespace llvm { ... }` blocks. Policy: emitters are updated to produce `LLVM_NAMESPACE_BEGIN / _END` directly; `.inc` files in the source tree that are hand-written keep the same treatment as any other source file. The rewriter does not modify generated `.inc` files; the TableGen emitter change is a dedicated sub-workstream of Stage 3b (batch B0, below) that must land before B1. Acceptance test: `rg '^namespace (llvm|mlir) \{' --glob '*.inc' <tree>` returns zero results in post-Stage-3 builds outside the §2 exclusion list.

**Bootstrap.** The rewriter is built from a pre-migration LLVM revision (upstream trunk) using an unmodified Clang, deployed into `../llvm-build/bin/clang-llvm-namespace-wrap`. It does not depend on the macros it emits. This breaks the chicken-and-egg.

#### 3b. Apply to the tree

Batches are defined by the transitive-include DAG (scanned programmatically via `clang-scan-deps` on the post-batch tree; directory layout below is the human-readable summary). Apply is **strictly sequential** in batch order; verification within each batch (build + test + symbol-diff) runs in parallel across the batch's sub-batches. The two-phase "parallel prep" from earlier drafts was unsound — a batch compiled with tagging-active but pulling unmigrated headers from a later batch produces mixed mangling in one TU.

| Batch | Scope | Rationale |
|---|---|---|
| B0 | `llvm/utils/TableGen/` emitters + MLIR tblgen emitters | Must land before B1 so generated `.inc` files emit the macros |
| B1 | `llvm/include/llvm/Support/` + `llvm/lib/Support/` | Root of the include DAG |
| B2 | `llvm/include/llvm/ADT/` | Builds on Support |
| B3 | `llvm/include/llvm/IR/`, `Analysis/`, `Transforms/`, `CodeGen/` + their `lib/` counterparts | Core IR |
| B4 | `llvm/tools/`, `llvm/unittests/` | Consumes B1–B3 |
| B5 | `mlir/include/mlir/` (minus §2 exclusions) + `mlir/lib/` (minus §2 exclusions) | MLIR core, depends on B1–B3 |
| B6 | `mlir/tools/`, `mlir/unittests/`, `mlir/test/` fixtures | MLIR consumers |
| B7 | All of `clang/` (sources and headers) minus §2 exclusions | Every top-level `namespace llvm { ... }` in `clang/`, not only forward-decl files |
| B8 | `lld/`, `lldb/`, `bolt/`, `polly/` consumers | Depends on B1–B7 |

Per-batch workflow:

1. Coordinator applies the rewriter to batch B_i on top of `main` (which already contains B_0 … B_{i-1}). Produces a single candidate branch `migrate/B_i`.
2. Coordinator splits `migrate/B_i` into sub-PRs by subdirectory or include-layer so no single PR exceeds ~500 files. Each sub-PR targets a single area-owner reviewer set. LLVM's area ownership (Support, ADT, IR, MLIR core, …) determines sub-PR granularity; expect 3-10 sub-PRs per batch for B3 and B5.
3. Verification subagents run in parallel, one per sub-PR, each in `../llvm-build-B_i-<sub>`:

```
Task: verify sub-PR <ID> of batch B_i.

Inputs:
  - source checkout at: <sub-PR branch>
  - build dir (subagent-private): ../llvm-build-B_i-<sub>
  - scope directories: <sub-PR scope>
  - §2 exclusion list
  - rewriter binary: ../llvm-build/bin/clang-llvm-namespace-wrap

Steps:
  1. Configure ../llvm-build-B_i-<sub> with -DLLVM_ABI_NAMESPACE=v23_0.
  2. Re-run the rewriter over the scope; diff against what the PR already
     contains. Non-empty diff = inconsistency, fail.
  3. Assert audit.json has zero entries under any §2-excluded path.
  4. Run the per-batch test suite (check-llvm / check-mlir / check-clang /
     check-lld / check-lldb as relevant to the batch scope). Test selection
     is tiered: per-sub-PR = touched components only; per-batch final =
     check-all on ../llvm-build.
  5. Run the §10 C API symbol-diff check (../llvm-build-B_i-<sub> vs
     ../llvm-build-untagged).
  6. Produce: diff, audit.json, build log, test log.

Output: verification report. Do NOT merge.
```

4. Coordinator merges sub-PRs in include-DAG order within the batch. After every merge, the coordinator re-runs the rewriter over the rest of the batch on the updated tree (catches new `namespace llvm { ... }` blocks that landed upstream during the batch's review window), updates the remaining sub-PRs, and re-dispatches verification.

5. After the final sub-PR of B_i merges, the coordinator runs `ninja -C ../llvm-build check-all` once as the batch-level gate.

**Iteration estimate.** Per-batch churn is empirical, not stipulated. Before B_i starts, the coordinator samples the previous 4 weeks of `git log --numstat` restricted to B_i's scope to forecast expected rebase iterations. Escalation threshold: if a batch exceeds 8 iterations, pause and re-plan (smaller sub-batches, shorter review window, or coordinated freeze window with area maintainers). Historical baseline for B3 / B5 scopes on LLVM's current merge rate: expect 3-6 iterations; budget accordingly.

#### 3c. Audit review

After each batch lands, the coordinator reviews aggregated `audit.json` entries. Each `kind` maps to a one-line rule and a test:

| `kind` | Rule | Test |
|---|---|---|
| `wrap_decl` | Must wrap outer `namespace llvm` or `namespace mlir` at file scope | Stage 1 unit test |
| `wrap_forward_decl` | Forward decl is now under the tag; downstream consumers must match | §10 test 5 (header/library mismatch) |
| `wrap_instantiation` | Explicit template instantiation resolves to tagged type | §10 test 7 (template instantiation) |
| `wrap_specialization` | `std::hash<llvm::X>` in `namespace std` resolves via inline | §10 test 8 (std specialization) |
| `wrap_friend` | Friend decl naming `llvm::X` resolves via inline | §10 test 9 (friend decl) |
| `wrap_out_of_class_def` | Out-of-class member definition resolves via inline | §10 test 10 (out-of-class def) |

`extern "C"` blocks that contained `namespace llvm` forward declarations (rare; e.g. in C++-only portions of `extern "C" { #ifdef __cplusplus ... }`) are left untouched by the rewriter and appear as `audit.json` entries with `kind = skipped_extern_c`. Coordinator decides whether to move the declaration out of `extern "C"` on a case-by-case basis.

#### 3d. Validation — MLIR `TypeID` / `-Bsymbolic`

This was "post-migration follow-up" in earlier drafts. Promoted to a Stage 3 exit gate because it could invalidate the MLIR side of the migration if tagging doesn't compose with `TypeID`.

- Reproduce the original `-Bsymbolic` / `TypeID` failure from PR51420 against untagged MLIR (establishes baseline).
- Re-run the same reproducer against tagged MLIR.
- Document the result: tagging either (a) lets us relax the `mlir/CMakeLists.txt:74-76` guard, (b) has no effect (guard stays), or (c) introduces a new failure mode (must be resolved before Stage 4). Outcome (c) blocks Stage 4.

**Stage 3 exit criteria:** with `../llvm-build` configured using `-DLLVM_ABI_NAMESPACE=v23_0`, `ninja -C ../llvm-build check-all` passes across `llvm`, `mlir`, `clang`, `lld`, `lldb`. The §10 symbol-diff guard passes on every batch and on the final merged tree. The §10 two-copies-static integration test demonstrates two distinct tags coexisting in one binary. The Stage 3d validation report is on file.

#### 3e. Rollback plan

Once a batch merges, setting `LLVM_ABI_NAMESPACE=` (empty) restores pre-migration *runtime behavior* (no tag in symbols) but does not revert the source edits — files now go through `LLVM_NAMESPACE_BEGIN / _END`, which expand to plain `namespace llvm {` under an empty tag. Full source rollback requires `git revert` of the batch's merge commits, which is a large but mechanical operation.

Rollback triggers: (a) Stage 3d validation outcome (c); (b) a Windows platform blocker discovered during integration (§10 test 3); (c) an irreparable review-capacity exhaustion for a batch.

The coordinator owns the rollback decision. Rollback of a landed batch cascades: all later batches that touched files in the reverted batch's scope must be rebased or re-applied.

### Stage 4 — Default enablement

- Change the CMake default for `LLVM_ABI_NAMESPACE` from empty to `v${LLVM_VERSION_MAJOR}_${LLVM_VERSION_MINOR}`.
- The empty-tag escape hatch (`-DLLVM_ABI_NAMESPACE=`) remains supported.

Stage 4 lands no earlier than the LLVM release cycle following Stage 3's completion. Between them, downstream consumers have at least one full release cycle with tagging opt-in to migrate their forward declarations. Release notes in the Stage 4 release enumerate: the default tag value, the empty-tag escape hatch, the `LLVMConfig.cmake` contract, the PCH/BMI invalidation requirement, and the `llvm/docs/InterfaceExportAnnotations.rst` cross-reference.

**Exit criteria:** with CMake defaults, `ninja -C ../llvm-build check-all` passes; `nm ../llvm-build/lib/libLLVM*.so | c++filt` shows tagged symbols; release notes have merged.

### Stage 5 — Tooling updates and enforcement

Each bullet dispatched as a separate subagent (invariant 8.0(1)); coordinator reviews.

- `llvm/utils/gdb-scripts/`, `llvm/utils/lldbDataFormatters.py`, LLDB type-summary patterns matching against `llvm::` — update to tolerate the inline-namespace component.
- `llvm-cxxfilt` / `__cxa_demangle` demangling output: verify the inline namespace renders readably in fully-qualified output and hidable under a flag matching libc++'s `__1` treatment.
- `llvm/utils/`: grep for hard-coded `llvm::` patterns in profilers, symbolizers, and perf-tooling scripts; produce a fix list.
- `llvm/docs/InterfaceExportAnnotations.rst` update: document the namespace-macro pattern, the CMake tag knob, the `LLVMConfig.cmake` contract, PCH/BMI cache-key guidance, and the composition rule (every exported declaration uses `LLVM_NAMESPACE_BEGIN / _END` around `LLVM_ABI`-annotated members).
- Rewriter lifecycle decision: retire or promote. The coordinator either (a) removes `clang-tools-extra/clang-llvm-namespace-wrap/` from the tree, or (b) keeps it wired to a CI check that fails PRs adding raw `namespace llvm { ... }` at file scope outside the §2 exclusion list. Recommended: (b), because it enforces the migration for new code at near-zero maintenance cost.
- Forward-declaration CI grep (v1 substitute for clang-tidy): a small script `llvm/utils/check-inline-namespace.py` runs on PRs touching `llvm/`, `clang/`, `mlir/`, `lld/`, `lldb/` headers and fails if it finds `^namespace (llvm|mlir) \{` (anchored to line start, file scope) outside the §2 exclusion list. This is not a replacement for (b) above but a cheaper backstop that runs on every PR.

## 9. Implementation details

### 9.1 Rewriter matcher (illustrative)

This snippet shows the shape, not the production matcher set. The real matcher covers nested namespaces, out-of-class definitions, friend declarations, and `namespace std` specializations; Stage 3a lit tests are authoritative.

```cpp
namespaceDecl(
    isExpansionInMainFile(),
    unless(isImplicit()),
    anyOf(hasName("llvm"), hasName("mlir")),
    hasParent(translationUnitDecl()))
  .bind("ns");
```

Rewriting uses `Replacements`:

- At the `namespace llvm` or `namespace mlir` token, substitute `LLVM_NAMESPACE_BEGIN` / `MLIR_NAMESPACE_BEGIN`.
- At the matching closing `}`, substitute `LLVM_NAMESPACE_END` / `MLIR_NAMESPACE_END`.
- Preserve any trailing `// namespace llvm` comment.
- For `namespace llvm::foo { ... }`, split into outer `LLVM_NAMESPACE_BEGIN namespace foo { ... } LLVM_NAMESPACE_END` so the inline insertion point sits at the outer `llvm`.

Invocation (`-p` argument is a directory, per clang's `CommonOptionsParser`):

```
../llvm-build/bin/clang-llvm-namespace-wrap \
  -p ../llvm-build-B_i \
  --include-path <batch-scope-dirs> \
  --exclude-path llvm/include/llvm-c/ \
  --exclude-path clang/include/clang-c/ \
  --exclude-path mlir/include/mlir-c/ \
  --exclude-path mlir/lib/CAPI/ \
  --audit-out audit.json \
  <source files>
```

### 9.2 Forward-declaration form

Before:

```cpp
namespace llvm {
class Value;
class Module;
}
```

After:

```cpp
#include "llvm/Support/ABINamespace.h"

LLVM_NAMESPACE_BEGIN
class Value;
class Module;
LLVM_NAMESPACE_END
```

Downstream consumers apply the same transformation to their own forward declarations when they upgrade to a tagged LLVM. Undetected, unmigrated declarations fail at link time with an undefined `llvm::vX_Y::Foo` symbol.

## 10. Testing

Tests are grouped by the design property each validates.

**Property: macro expansion is ABI-neutral when tagging is inactive.**

1. **Mangled-name snapshot + cross-TU type identity** (`unittests/Support/ABINamespaceTest.cpp`). Two fixture TUs define and use `llvm::TestType`. Assert:
   - With tagging inactive: both TUs produce the same mangled name (`_ZN4llvm8TestTypeE`); `typeid(TestType)` equality across TUs; `dynamic_cast` works across TUs.
   - With `LLVM_ABI_NAMESPACE=v23_0`: same properties with mangled name `_ZN4llvm5v23_08TestType…`.
   - Bit-identical **exported-symbol set** (from `nm --defined-only --extern-only`, filtered to C++ linkage and sorted) between "tagging inactive via empty CMake cache variable" and "tagging inactive via no CMake variable at all." This is the precise statement; whole-object-file byte-identity is not claimed.

**Property: the C API is untouched.**

2. **C API symbol-diff guard** (gates every Stage 3 sub-PR and every main CI build). Compare `../llvm-build-untagged` and `../llvm-build`. Extract C-linkage symbols using platform-specific tooling, not demangler round-trips:
   - ELF: `readelf -Ws` filtered to `STB_GLOBAL` / `STB_WEAK` with `STT_FUNC` / `STT_OBJECT` / `STT_NOTYPE`; exclude compiler-runtime allowlist (`__cxa_*`, `_Unwind_*`, `__gxx_*`, guard variables per Itanium C++ ABI §3.3.2).
   - Mach-O: `nm -gU` with the same allowlist.
   - PE/COFF: `dumpbin /exports` on the `.dll` + allowlist.
   Normalize (sort, deduplicate). The two lists must match exactly. Any delta fails the build.

**Property: two copies coexist.**

3. **Two-copies static-link integration test** (`llvm/test/Integration/ABINamespace/two-copies-static/`). Build two `libLLVMSupport.a`, one with `-DLLVM_ABI_NAMESPACE=v23_0_a`, one with `v23_0_b`. Link both into one `main.cpp` via two wrapper TUs exercising the **unqualified** `llvm::` API (the source-level spelling users actually write), not hand-qualified `llvm::v23_0_a::`. Verify each wrapper operates on its own copy's state (each has its own `cl::opt` registry, `ManagedStatic`, `TypeID` roots). **Gating per platform:**
   - Linux: blocking (Stage 3 exit criterion).
   - macOS: blocking.
   - Windows: best-effort in v1; non-blocking but runs on CI. Tracked for v2 promotion.

4. **Two-copies dlopen test.** Same as #3 but via `dlopen` / `LoadLibrary`. Linux + macOS blocking; Windows deferred to a v2 follow-up once `LLVM_BUILD_LLVM_DYLIB` on Windows stabilizes.

**Property: header/library mismatch fails loudly.**

5. **Header/library tag mismatch test.** Library built with `LLVM_ABI_NAMESPACE=a`, consumer with `LLVM_ABI_NAMESPACE=b`. Expect `ld` failure with a message naming `llvm::a::` (or `llvm::b::`, depending on direction). Platforms: Linux blocking; macOS blocking; Windows best-effort.

**Property: per-copy singletons.**

6. **`cl::opt` dual-registration test** (folded into #3). Each copy registers a distinct `cl::opt`; `ParseCommandLineOptions` on one copy sees its own flags only.

**Property: design invariants for template / specialization / friend / out-of-class cases.** These promote §3c's "spot-check" to dedicated tests.

7. **Template instantiation test.** `template class llvm::SmallVector<int>;` in a source file under tagging-active; `nm` shows the instantiation under the tag.
8. **`std::hash` specialization test.** `std::hash<llvm::StringRef>` in `namespace std`; invocation from a tagging-active TU resolves, mangled symbol contains the tag on `llvm::StringRef`.
9. **Friend declaration test.** `class Foo { friend llvm::X; };` in a tagging-active TU compiles; `llvm::X` resolves to the tagged type.
10. **Out-of-class member definition test.** `void llvm::Foo::bar() {}` in a `.cpp` file under tagging-active resolves and mangles under the tag.

## 11. Risks

### Per-copy `cl::opt` and `ManagedStatic`

Each tagged copy has its own `cl::opt` registry, its own `ManagedStatic<PassRegistry>`, its own timer and stats infrastructure. This is the intended behavior: it is what fixes the double-registration abort. A tool wanting unified `--help` across two LLVMs must forward manually between registries. Downstream code that assumed process-wide-singleton behavior from these APIs (rare but it happens) requires adjustment.

### Inlined accessors from mismatched headers

Described in §5. Link-time failure catches out-of-line symbol references; inline-only mismatches remain undetectable at runtime in v1. Consumers must rebuild against matching headers; `LLVMConfig.cmake` exports the tag value that the build system is expected to enforce.

### Windows

Versioned namespaces work on PE/COFF — the tag is applied at name mangling, before the linker. Two caveats:
- MSVC mangling historically had corner-case bugs around inline namespaces; validated via §10 test 3 (Windows best-effort in v1, blocking in v2).
- `LLVM_BUILD_LLVM_DYLIB` on Windows is a separate unstable story that this plan does not unblock but whose eventual resolution is simplified by tagging (unique DLL export names).

### Demangled output noise

Stack traces, profiler output, and compiler error messages now contain `llvm::v23_0::` instead of `llvm::`. Tooling can hide the inline-namespace segment — libc++'s `__1` is already handled that way by LLDB and modern demanglers. Stage 5 covers the LLVM-adjacent subset.

### Downstream forward declarations

Every project that forward-declares LLVM types outside `llvm/` headers must migrate to `LLVM_NAMESPACE_BEGIN / LLVM_NAMESPACE_END`. Signal: link error naming `llvm::vX_Y::Foo`. Stage 5 ships a grep-based CI check as a cheap in-tree backstop; downstreams rely on link failure.

### PCH / BMI / module caches

The tag is a preprocessor-visible macro in a generated header. Any PCH or C++20 BMI consumed across tag boundaries is an ODR violation the linker cannot diagnose. Build systems that cache these artifacts must include the tag in the cache key; LLVM itself rebuilds from scratch on tag change because the CMake propagation re-triggers the full graph via the generated `ABINamespaceTag.h`.

### Tag collision across vendors

Two downstream vendors can pick the same `LLVM_ABI_NAMESPACE` override value and collide while believing they had isolation. Convention: vendors use a prefixed tag (`v23_0_acme`), documented in the `LLVMConfig.cmake` export and in the Stage 5 RST updates.

### Review capacity

Batches B3 (core IR) and B5 (MLIR core) will generate many sub-PRs requiring coordinated review by area owners across Support, ADT, IR, MLIR. Stage 3b assumes at least one nominated area-owner reviewer per sub-PR. Without that, the serialized-merge loop stalls at review, not at test. The coordinator is expected to line up reviewers ahead of dispatching sub-PRs for a batch.

## 12. Out of scope

- **C API versioning.** Needs separate mechanism (ELF symbol versioning, renamed public symbols, or parallel installations).
- **Runtime tag self-report.** Deferred; see §5 entry for rationale (stable probe would require new C API surface).
- **Header-level ABI-signature hashing** (à la `__GLIBCXX_ABI_TAG`) to make same-tag-different-layout mismatches loud.
- **Cross-copy `cl::opt` registry unification** beyond manual forwarding.
- **LLVM_ABI annotation change.** The annotations stay as they are; their composition with the namespace macros is documented in Stage 5 RST updates.

## 13. References

- `llvm/include/llvm/Support/Compiler.h` lines 141-154: `LLVM_LIBRARY_VISIBILITY` / `LLVM_LIBRARY_VISIBILITY_NAMESPACE` / `LLVM_ALWAYS_EXPORT`.
- `llvm/include/llvm/Support/Compiler.h` lines 156+: `LLVM_ABI` / `LLVM_ABI_EXPORT` / `LLVM_TEMPLATE_ABI`.
- `llvm/docs/InterfaceExportAnnotations.rst`: existing ABI annotation guide (updated in Stage 5).
- `mlir/CMakeLists.txt` lines 74-76: `-Bsymbolic[^-]` / `TypeID` fatal-error guard; PR51420.
- `cmake/Modules/LLVMVersion.cmake`: `LLVM_VERSION_MAJOR` / `LLVM_VERSION_MINOR` source of truth.
- Prior art: libstdc++ `_GLIBCXX_USE_CXX11_ABI` / `std::__cxx11` inline namespace.

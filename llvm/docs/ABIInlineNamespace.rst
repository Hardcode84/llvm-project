LLVM Inline Versioned Namespace (ABI Isolation)
================================================

.. contents::
   :local:

Overview
--------
The LLVM build system can wrap C++ declarations under ``namespace llvm {
... }`` in an *inline versioned namespace* whose tag is controlled by the
``LLVM_ABI_NAMESPACE`` CMake variable.  MLIR has a parallel
``MLIR_ABI_NAMESPACE`` that by default tracks LLVM's value.

When the tag is active, ``namespace llvm { ... }`` expands to::

   namespace llvm { inline namespace v23_0 { ... } }

so every *tagged* mangled C++ name acquires an extra substitution component
(``_ZN4llvm5v23_0...``).  Source-level names resolve unchanged through the
``inline`` keyword: ``llvm::Value`` still refers to ``llvm::v23_0::Value``
on both the declaring and using side.

The goal is to let two independently-built copies of the LLVM/MLIR C++ API
coexist in a single process without colliding on mangled C++ symbols — for
example, a vendored compiler plugin built against ``libLLVM-23.0`` loaded
into a host application that also links its own ``libLLVM-23.0``.

Scope
~~~~~
Tagging covers the bulk of the LLVM C++ API: IR, ADT, Support (except a
small number of headers listed below), Analysis, CodeGen, Transforms,
TableGen-generated code, MLIR, Clang AST, lld, flang, and the tools that
link directly against these libraries.  The same applies to MLIR's C++
API under ``namespace mlir``.

The following zones are **intentionally untagged** and will NOT be
isolated across builds with distinct tags:

* The **C API** (``llvm-c/``, ``mlir-c/``, any ``extern "C"`` surface).
  C symbol names have no mangling and are shared process-globally.
* The **Demangle library** (``llvm/include/llvm/Demangle``,
  ``llvm/lib/Demangle``).  Needed for external symbolicators that expect
  stable cross-version names.
* The **profile and coverage data format libraries**
  (``llvm/include/llvm/ProfileData``, ``llvm/lib/ProfileData``,
  ``llvm/tools/llvm-profgen``, parts of ``llvm/tools/llvm-cov``, and
  selected consumer headers in ``llvm/include/llvm/Analysis`` and
  ``llvm/include/llvm/Transforms``).  These share byte-identical ``.inc``
  headers with the compiler-rt profiling runtimes that cannot depend on
  ``LLVMSupport``.
* **compiler-rt** in its entirety (runtime libraries that do not link
  against ``LLVMSupport`` at all).

See `ABI-stable untagged zones`_ for the detailed list and rationale.

.. note::

   **Tagging alone is not sufficient for dual-LLVM coexistence.** Running
   two LLVMs in one process also requires distinct ``DT_SONAME``
   identifiers (or static linkage, or a carefully-scoped ``RUNPATH``),
   plus an awareness of the untagged zones above.  See
   `Distribution / ELF considerations`_.

Default behavior
----------------
Upstream's default tag is derived from the LLVM version::

   LLVM_ABI_NAMESPACE = v${LLVM_VERSION_MAJOR}_${LLVM_VERSION_MINOR}

so a released LLVM 23.0 ships tagged C++ symbols under ``llvm::v23_0``.
Consequences:

* Two different LLVM minor versions produce different tagged C++ symbol
  names.  If you meet the other dual-LLVM prerequisites
  (`Distribution / ELF considerations`_) their C++ surfaces stay
  separate at link and ``dlopen`` time.

* Distributors are strongly encouraged to *further suffix* the default to
  distinguish their builds from upstream's and from other vendors'.

* Setting ``LLVM_ABI_NAMESPACE=""`` explicitly disables tagging.  The
  expanded code is identical to pre-tag releases; the mode exists for
  consumers that need bit-identical mangling (for example, downstream
  tooling that greps by raw symbol name, or strict ABI archaeology).

Configuration
-------------
``LLVM_ABI_NAMESPACE``
    Identifier for the inline namespace.  Defaults to
    ``v${LLVM_VERSION_MAJOR}_${LLVM_VERSION_MINOR}``.  Empty disables
    tagging.  Must match the C identifier grammar
    ``^[A-Za-z_][A-Za-z0-9_]*$`` when set; CMake errors out otherwise.

``MLIR_ABI_NAMESPACE``
    Identifier for MLIR's inline namespace.  Defaults to the current
    ``LLVM_ABI_NAMESPACE`` value.  Override independently only if you
    know MLIR/LLVM types never cross the boundary in your use case;
    diverging tags is otherwise unsupported for generic embedding.

The value flows into built artifacts through
``llvm/Support/ABINamespaceTag.h`` (generated from
``ABINamespaceTag.h.cmake``) and is re-exported to downstream CMake
consumers as the read-only ``LLVM_ABI_NAMESPACE`` variable in
``LLVMConfig.cmake``.

Distributors
~~~~~~~~~~~~
A vendor shipping a compiler that embeds LLVM should pick a vendor-
specific suffix::

   cmake -DLLVM_ABI_NAMESPACE=v23_0_acme \
         -DMLIR_ABI_NAMESPACE=v23_0_acme ...

See `Distribution / ELF considerations`_ for the full set of steps a
real dual-LLVM distribution needs to take.

Embedders
~~~~~~~~~
If you embed LLVM as a C++ API consumer, you do not need to change your
code.  ``llvm::Value``, ``mlir::Type``, and every other public type
continue to work exactly as before thanks to ``inline namespace``.  The
only observable effects are in symbol dumps, debuggers, and linker
diagnostics.

.. warning::

   Do not redefine ``LLVM_ABI_NAMESPACE`` on your own compile line.  The
   tag that matches the installed libraries comes from
   ``llvm/Support/ABINamespaceTag.h``, which sits next to the installed
   headers.  Overriding the macro on your compile line produces object
   files whose references point at symbols that do not exist in the
   libraries you link against.

   If your build system maintains precompiled-header or C++20 BMI caches,
   include ``LLVM_ABI_NAMESPACE`` in the cache key — a cache entry
   produced with a different tag is an ODR violation the linker cannot
   diagnose.

Distribution / ELF considerations
---------------------------------
Tagging is one of three ingredients needed for real dual-LLVM
coexistence on ELF/``dlopen`` platforms.  A plugin that must load into a
process that already has a different LLVM copy resident should also:

1. **Use a distinct SONAME.**  Build the vendored LLVM with a SONAME
   that differs from the system package.  Without this the dynamic
   linker reuses a single mapping for both consumers and the plugin's
   tagged symbols fail to resolve::

       cmake -DLLVM_ABI_NAMESPACE=v23_0_acme \
             -DCMAKE_SHARED_LINKER_FLAGS="-Wl,-soname,libLLVM-23.0-acme.so" ...

2. **Use symbolic binding and a private RUNPATH.**  Link the plugin with
   ``-Wl,-Bsymbolic`` (or equivalent) and set its ``RUNPATH`` to the
   vendored LLVM directory so internal calls resolve locally before the
   dynamic linker searches the global scope.

3. **Statically link the untagged zones.**  If your plugin uses
   ``LLVMDemangle``, ``LLVMProfileData``, or the C API, prefer static
   linkage.  The untagged symbols are then embedded into the plugin
   with hidden visibility and cannot collide with the host's copy.  If
   static linkage is not an option, accept that these symbols are shared
   and design around it (for example, confine profile data work to one
   of the two LLVMs).

Without (1), the tag never comes into play because there is only one
mapping to begin with.  Without (2), the host process's copy shadows the
plugin's.  Without (3), the untagged surface is shared despite tagging.

Example Acme distro configuration::

    cmake -G Ninja \
      -DLLVM_ABI_NAMESPACE=v23_0_acme \
      -DMLIR_ABI_NAMESPACE=v23_0_acme \
      -DCMAKE_SHARED_LINKER_FLAGS="-Wl,-soname,libLLVM-23.0-acme.so -Wl,-Bsymbolic" \
      ...

C API
-----
The LLVM-C and MLIR-C APIs are **not** tagged.  Functions in ``extern
"C"`` blocks, headers under ``include/llvm-c/`` and ``include/mlir-c/``,
and any translation unit that is part of a C-API library are never
wrapped in ``LLVM_NAMESPACE_BEGIN/END``.

What tagging gives the C API:

* Stable symbol names across tagged and untagged builds.  A client that
  calls ``LLVMContextCreate`` sees the same mangled-free name at every
  version; ``dlsym("LLVMContextCreate")`` keeps working across tagged,
  untagged, and differently-tagged builds.

* Freedom from ``_ZN4llvm3FooEv vs _ZN4llvm5v23_0...`` link errors: the
  C API does not expose mangled C++ symbols on its public surface.

What tagging does **not** give the C API:

* Runtime isolation of two LLVMs in one process.  ``LLVMContextCreate``
  is a plain global symbol; if two LLVM DSOs are loaded together, the
  dynamic linker resolves every call to whichever copy the linker bound
  first.  The C API is therefore a *single-LLVM* interface, no matter
  how many LLVM builds are present.

Rule of thumb: **prefer the C API for cross-build stability of symbol
names**; **use distinct SONAMEs for cross-build runtime isolation**.  The
two mechanisms solve different problems and are both needed for a
dual-LLVM distribution.

ABI-stable untagged zones
-------------------------
The zones below are deliberately kept in raw ``namespace llvm {`` and
are exempted from the CI backstop (``llvm/utils/check-abi-namespace.py``).
Each zone has a specific stability contract that tagging would break.

Demangle library
~~~~~~~~~~~~~~~~
``LLVMDemangle`` (``llvm/include/llvm/Demangle``,
``llvm/lib/Demangle``) is used by external symbolicators — ``perf``,
``llvm-objdump``, LLDB's embedded plugins, ``libc++abi`` — that expect
stable cross-version names for ``llvm::itaniumDemangle``,
``llvm::microsoftDemangle``, and friends.  It is also designed to be
standalone: it does not link against ``LLVMSupport`` and is sometimes
vendored directly by consumers.

Its internal namespace is controlled by ``DEMANGLE_NAMESPACE_BEGIN`` in
``DemangleConfig.h``, not by ``LLVM_NAMESPACE_BEGIN``.  The two macro
families are intentionally separate.

**Coexistence caveat:** ``LLVMDemangle`` is linked into ``libLLVM``.  Two
tagged ``libLLVM`` DSOs loaded into one process still export duplicate
untagged Demangle symbols.  For demangle isolation, link
``LLVMDemangle`` statically into your plugin or vendor the library.

Profile and coverage data format libraries
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The ProfileData subsystem shares byte-identical ``.inc`` headers with
the compiler-rt profiling runtimes (``MemProfData.inc``,
``InstrProfData.inc``, ``CtxInstrContextNode.h``).  compiler-rt does not
link against ``LLVMSupport`` and cannot include the namespace macros,
so these headers must use raw ``namespace llvm { ... }`` to keep the two
halves in sync on disk.

The following files and directories are untagged as a result:

* ``llvm/include/llvm/ProfileData/`` — all headers and ``.inc`` files.
* ``llvm/lib/ProfileData/`` — bodies that define ProfileData types.
* ``llvm/tools/llvm-profgen/`` — directly references sampleprof/memprof
  types and must match their namespace.
* ``llvm/tools/llvm-cov/CoverageFilters.h`` — references
  ``llvm::coverage`` types at file scope.
* ``llvm/include/llvm/Support/Discriminator.h`` — defines
  ``llvm::sampleprof::Discriminator`` which is shared with ProfileData.
* ``llvm/include/llvm/Analysis/MemoryProfileInfo.h``,
  ``llvm/include/llvm/Analysis/StaticDataProfileInfo.h``,
  ``llvm/include/llvm/Transforms/Instrumentation/MemProfUse.h``,
  ``llvm/include/llvm/Transforms/IPO/ProfiledCallGraph.h`` — consumer
  headers that forward-declare ProfileData types or expose
  ``cl::opt`` externs whose definitions live in untagged ProfileData.

**Coexistence caveat:** types such as ``llvm::InstrProfError``,
``llvm::sampleprof::FunctionSamples``, ``llvm::coverage::CoverageMapping``,
and ``cl::opt`` options like ``EnableVTableProfileUse`` are untagged and
can collide if two LLVMs are co-loaded.  If your plugin uses the
ProfileData C++ API, prefer static linkage of ``LLVMProfileData`` into
your plugin, or confine profile/coverage work to a single LLVM.

Referring to core LLVM types from an untagged ProfileData header
requires a small pattern described in `Referring to symbols across the
tagged boundary`_.

compiler-rt
~~~~~~~~~~~
``compiler-rt`` is a family of runtime libraries (sanitizers, profiling,
builtins) that do not link against ``LLVMSupport`` at all.  Its handful
of ``namespace llvm { ... }`` blocks are either shared on-disk data
formats or sanitizer test helpers that build without LLVM includes.  The
tree is excluded from the rewriter, from ``Compiler.h``'s macro include,
and from the backstop.

Macros
------
Headers that used to spell ``namespace llvm { ... }`` now use macros
defined in ``llvm/include/llvm/Support/ABINamespace.h`` (automatically
included via ``llvm/Support/Compiler.h``):

``LLVM_NAMESPACE_BEGIN`` / ``LLVM_NAMESPACE_END``
    File-scope namespace wrappers.  Replace every raw ``namespace llvm
    {`` and matching ``}`` at file scope.

``LLVM_ABI_NS``
    Nested-name-specifier that resolves to ``llvm::vX_Y`` when the tag is
    active and plain ``llvm`` when it is not.  Use it at the start of
    any **out-of-class** definition that would otherwise spell
    ``llvm::Foo`` at file scope, for example::

        struct LLVM_ABI_NS::Foo { int x; };                 // was: struct llvm::Foo
        void LLVM_ABI_NS::Foo::bar() {}                     // was: void llvm::Foo::bar()
        template class LLVM_ABI_NS::SmallVectorBase<int>;   // was: template class llvm::...

    This is required because the Itanium C++ ABI mangles such
    definitions by the **spelled** qualifier, not by name lookup.
    Without ``LLVM_ABI_NS``, a tagged build would mangle the definition
    as ``llvm::Foo`` and fail to resolve against the declaration in
    ``llvm::v23_0::Foo``.

MLIR has the parallel ``MLIR_NAMESPACE_BEGIN``, ``MLIR_NAMESPACE_END``,
and ``MLIR_ABI_NS`` macros in ``mlir/include/mlir/Support/ABINamespace.h``.

.. note::

   ``LLVM_ABI`` (``llvm/include/llvm/Support/Compiler.h``) and
   ``LLVM_ABI_NS`` are **unrelated**.  ``LLVM_ABI`` is a visibility
   annotation (``__attribute__((visibility("default")))`` equivalent)
   that marks a symbol as part of the public shared-library ABI.
   ``LLVM_ABI_NS`` is the inline-namespace qualifier described above.
   The names share the ``LLVM_ABI`` prefix for historical reasons.

Authoring guidance
------------------
When adding new LLVM/MLIR C++ code:

1. **Use the macros.** ``LLVM_NAMESPACE_BEGIN`` at the outermost
   namespace opening, ``LLVM_NAMESPACE_END`` at its close.  Keep nested
   namespaces as ordinary ``namespace detail { ... }`` inside.

2. **Do not hand-write ``namespace llvm {``** at file scope outside the
   documented untagged zones.  The rewriter and the CI backstop
   (``llvm/utils/check-abi-namespace.py``) flag such cases.

3. **Qualify out-of-line definitions with ``LLVM_ABI_NS::``.** Member
   function definitions, explicit template instantiations, DenseMapInfo
   specializations, ``DOTGraphTraits`` specializations, and similar
   constructs must use the macro-qualified form.

4. **TableGen emitters** generate code through
   ``llvm::NamespaceEmitter``
   (``llvm/include/llvm/TableGen/CodeGenHelpers.h``), which emits
   ``LLVM_NAMESPACE_BEGIN/END`` or ``MLIR_NAMESPACE_BEGIN/END``
   whenever the target namespace starts with ``llvm`` or ``mlir``.  New
   emitters should use this helper rather than emitting raw
   ``namespace X {`` strings.

Referring to symbols across the tagged boundary
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Code that lives in an untagged zone sometimes needs to refer to symbols
that live in a tagged zone (for example, a forward declaration of
``llvm::raw_ostream`` inside an untagged ProfileData header).  The rule
is: **the namespace you spell must match the namespace the symbol was
defined in.**

For a tagged declaration from an untagged header, wrap the
forward-declaration in ``LLVM_NAMESPACE_BEGIN/END`` and keep the body
raw:

.. code-block:: c++

   // Untagged header: llvm/include/llvm/ProfileData/SampleProf.h
   LLVM_NAMESPACE_BEGIN
   class raw_ostream;       // tagged fwd-decl of tagged llvm::v23_0::raw_ostream
   class DILocation;
   LLVM_NAMESPACE_END

   namespace llvm {         // untagged, matches ProfileData's ABI zone
   namespace sampleprof {
   // ... ProfileData API ...
   } // namespace sampleprof
   } // namespace llvm

Similarly, an ``extern`` declaration for a ``cl::opt`` must match the
namespace of its definition:

.. code-block:: c++

   // Option defined in untagged ProfileData (llvm/lib/ProfileData/InstrProf.cpp):
   namespace llvm {
   extern cl::opt<bool> EnableVTableProfileUse;
   } // namespace llvm

   // Option defined in tagged code (llvm/lib/Support/CommandLine.cpp):
   LLVM_NAMESPACE_BEGIN
   LLVM_ABI extern cl::opt<bool> SomeTaggedOption;
   LLVM_NAMESPACE_END

Getting this wrong produces a linker error whose mangled form tells you
which side is in the wrong namespace; see `Diagnosing link errors`_.

Diagnosing link errors
----------------------
The most common authoring mistake — forgetting ``LLVM_ABI_NS::`` on an
out-of-line definition, or putting an ``extern`` in the wrong namespace
— surfaces as an ``undefined reference`` whose mangled form tells you
exactly which side is wrong.

Example: an out-of-line definition spelled ``void llvm::Foo::bar() {}``
in a tagged build where the declaration lives in the tagged namespace:

.. code-block:: text

    undefined reference to `llvm::v23_0::Foo::bar()'

Demangle both the referenced and the defined symbol to compare::

    $ echo _ZN4llvm5v23_03Foo3barEv | llvm-cxxfilt
    llvm::v23_0::Foo::bar()
    $ echo _ZN4llvm3Foo3barEv | llvm-cxxfilt
    llvm::Foo::bar()

If the reference has ``v23_0`` but the definition doesn't, the
definition is missing ``LLVM_ABI_NS::``.  If the reference doesn't have
``v23_0`` but the definition does, the referring code opened raw
``namespace llvm {`` where it should have used ``LLVM_NAMESPACE_BEGIN``.

The mirror case for MLIR is identical with ``mlir::`` / ``MLIR_ABI_NS``
/ ``MLIR_NAMESPACE_BEGIN``.

CI backstop
-----------
``llvm/utils/check-abi-namespace.py`` is a grep-level scanner that flags
raw ``namespace llvm {`` / ``namespace mlir {`` openings and raw
``struct llvm::Foo``-style out-of-line definitions outside the
documented untagged zones.  Run it from the monorepo root::

    python3 llvm/utils/check-abi-namespace.py --root .

Exit status:

* 0 — no violations.
* 1 — at least one violation.
* 2 — invalid invocation.

The script maintains an explicit exclusion list that mirrors the
`ABI-stable untagged zones`_ section.  Any divergence between this doc
and the script's ``EXCLUDE_DIRS`` is a bug in whichever is out of date.

.. note::

   Upstream CI does not yet run this script automatically.  Vendors and
   downstreams are encouraged to wire it into their pre-merge pipelines.
   The authoritative list of untagged zones lives both here and in the
   script's exclusion list; keep them synchronised.

The scanner is a complement, not a replacement, for the rewriter
(``llvm/utils/abi-namespace-wrap/abi_namespace_wrap.py``).  It does not
parse C++; its ``{``/``}`` depth counter does not see
``LLVM_NAMESPACE_BEGIN`` / ``LLVM_NAMESPACE_END`` as openers, which is
accurate enough for top-level detection but means it cannot diagnose
macro/raw-namespace mixing inside a file.

Debugging tagged builds
-----------------------
The tag appears in tool output:

* ``nm -C`` and ``c++filt`` render the demangled name as
  ``llvm::v23_0::Foo``.
* ``gdb`` and ``lldb`` accept both ``llvm::Foo`` and ``llvm::v23_0::Foo``
  in breakpoint commands because inline namespaces participate in name
  lookup.
* Libraries like ``libLLVMSupport.a`` contain tagged symbols for
  tagged LLVM entities (verified by
  ``llvm/unittests/Support/ABINamespaceTest.cpp``).

To sanity-check symbol mangling after a build::

    nm -g --defined-only $BUILD/lib/libLLVMSupport.a \
      | grep _ZN4llvm | grep -v _ZN4llvm5v23_0

should report nothing for LLVM entities in the tagged zone.  (The
ProfileData and Demangle libraries are expected to report untagged
``_ZN4llvm`` symbols; see `ABI-stable untagged zones`_.)

C API preservation check
~~~~~~~~~~~~~~~~~~~~~~~~
C API symbols live in their own namespaces (``LLVMFoo``, ``mlirFoo``)
and remain untagged.  Verify with::

    nm -g --defined-only $BUILD/lib/libLLVMCore.a | grep '^T LLVM' | head

The output should contain raw ``T LLVMContextCreate`` and friends — not
``T _ZN4llvm5v23_0...``.

Demangler and debugger integration
----------------------------------
The inline versioned namespace is a normal C++ inline namespace from the
point of view of every tool in the toolchain — no special demangler,
pretty-printer, or debugger plugin is required.

c++filt / llvm-cxxfilt
~~~~~~~~~~~~~~~~~~~~~~
The default Itanium demangler emits the full, tagged name::

    $ echo _ZN4llvm5v23_05ValueC1ENS0_4TypeE | llvm-cxxfilt
    llvm::v23_0::Value::Value(llvm::v23_0::Type)

Use ``--no-params`` to omit argument types for a more compact form.  The
``v23_0`` component always appears on tagged C++ symbols; C API symbols
(``LLVM*`` / ``mlir*``) and symbols from the untagged zones never carry
it.

If you specifically want to strip the version substring for grep-
compatibility with pre-tag releases, a sed post-filter works::

    llvm-cxxfilt < symbols.txt | sed -E 's/(llvm|mlir)::v[0-9]+_[0-9]+[_A-Za-z0-9]*:://g'

This is a purely cosmetic step for human-readable log scraping; do
**not** use it on symbols that are about to be fed into a build, linker,
or ``dlsym`` call.

gdb
~~~
GDB treats an inline namespace exactly as the C++ standard requires:
names are first looked up as if the inline namespace were transparent,
so all of the following work and are equivalent::

    (gdb) break llvm::Value::setName
    (gdb) break llvm::v23_0::Value::setName
    (gdb) rbreak ^llvm::v23_0::Value::

Stack traces print the tagged name::

    #0  0x... in llvm::v23_0::raw_ostream::write (this=..., Ptr=..., Size=...)
        at llvm/lib/Support/raw_ostream.cpp:...

If you are attaching to a core file produced by one build and loading
symbols from a differently-tagged build, gdb will not find a match.
This is the expected outcome: two builds with distinct tags expose
distinct C++ ABIs.

lldb
~~~~
LLDB's SBType and the command-line ``breakpoint`` / ``expression``
interpreters are inline-namespace aware.  Short-form names resolve
through the inline namespace::

    (lldb) b llvm::Value::setName
    (lldb) expr ((llvm::StringRef) ref).size()

When LLDB is itself built with ``LLVM_ABI_NAMESPACE`` active, its
internal ``lldb_private::`` symbols are unaffected (LLDB does not tag
its own namespaces).  Only the embedded LLVM/MLIR libraries carry the
tag.

Profiling / perf
~~~~~~~~~~~~~~~~
``perf report``, FlameGraphs, and similar tools display the tagged
demangled name the way ``c++filt`` does.  When aggregating profiles
across versions, strip the version component with the sed filter above
before bucketing by symbol name; otherwise symbols from LLVM 23.0 and
23.1 will appear as distinct entries even when they come from the same
source-level function.

Core dumps
~~~~~~~~~~
A core dump captures mangled symbols as they appear in the binary.
Analysing it requires the exact set of LLVM/MLIR libraries used at
crash time, including their tag.  The tag is a *compile-time*
preprocessor define (see ``llvm/Support/ABINamespaceTag.h``); it is not
stamped as a runtime string by default.

Record the tag in your distribution's build metadata (the
``LLVM_ABI_NAMESPACE`` CMake variable) and retrieve it from any shipped
library with ``nm``::

    $ nm -g --defined-only libLLVMSupport.so \
        | grep ' _ZN4llvm' | head -1 | c++filt
    llvm::v23_0::<some symbol>

The ``v23_0`` component of any tagged C++ symbol is the tag — the
build's identity from LLVM's point of view.

Limitations of ``nm``-based extraction:

* Stripped binaries lose the symbols that carry the tag; use unstripped
  archives or a ``.debug`` split.
* LTO builds may mangle the tag differently depending on toolchain
  version; verify with the same LTO mode used at crash time.
* On platforms without ``nm`` (Windows), ``llvm-readobj --symbols`` or
  ``dumpbin /symbols`` produce equivalent output.

Distributors who need a runtime-readable tag are free to add a small
``const char kLLVMABINamespace[] = "v23_0_acme";`` constant derived
from ``LLVM_ABI_NAMESPACE`` in their packaging repository; the upstream
tree keeps the tag macro-only to avoid per-TU string constants.

Related design decisions
------------------------
* ``-fvisibility=hidden`` alone does not isolate two LLVM copies; the
  tag complements it.
* The tag is an inline namespace (rather than a plain outer namespace)
  so ``llvm::`` remains usable source-level; only the mangled form
  changes.
* Header-only templates that consumers specialize for their own types
  keep working because the primary template and its specialization see
  the same ``llvm`` qualifier and the same inline namespace.
* TypeID and ``dyn_cast`` machinery is unaffected for tagged types:
  both rely on per-translation-unit static storage that is uniqued by
  tagged class identity, not by raw symbol name.  The same *is not*
  guaranteed across the tagged/untagged boundary — treat the untagged
  zones as independent RTTI islands.

Migration history
-----------------
* Stage 1 (LLVM 23): Introduced the ``LLVM_NAMESPACE_BEGIN/END`` and
  ``LLVM_ABI_NS`` macros and the ``LLVM_ABI_NAMESPACE`` CMake variable.
* Stage 2: Wired the generated ``ABINamespaceTag.h`` through
  ``Compiler.h`` and exported the tag via ``LLVMConfig.cmake``.
* Stage 3: Applied ``LLVM_NAMESPACE_BEGIN/END`` tree-wide via
  ``llvm/utils/abi-namespace-wrap/abi_namespace_wrap.py``; taught
  TableGen emitters to emit the macros; fixed out-of-line definitions
  to use ``LLVM_ABI_NS::``; validated the full build and test suites
  with the tag active.
* Stage 4: Flipped the CMake default from empty to
  ``v${MAJOR}_${MINOR}``.
* Stage 5: Documented the tag, carved out the ABI-stable untagged
  zones (Demangle, ProfileData, compiler-rt), and added the CI
  backstop script.

LLVM Inline Versioned Namespace (ABI Isolation)
================================================

.. contents::
   :local:

Overview
--------
Every C++ symbol under ``namespace llvm`` is wrapped in a per-build inline
namespace whose tag is controlled by the ``LLVM_ABI_NAMESPACE`` CMake
variable.  MLIR has a parallel ``MLIR_ABI_NAMESPACE`` that by default
tracks LLVM's value.  The purpose is to let two independently-built copies
of the LLVM or MLIR C++ API coexist inside a single process without
symbol collisions.

When the tag is active, ``namespace llvm { ... }`` becomes::

   namespace llvm { inline namespace v23_0 { ... } }

Mangled names acquire an extra ``5v23_0`` substitution component
(``_ZN4llvm5v23_0...``) so the symbols from two distinct builds — each
with its own tag — never collide at link or dlopen time.  Source-level
names resolve unchanged via the ``inline`` keyword: ``llvm::Value`` still
refers to ``llvm::v23_0::Value`` on both the declaring and the using side.

Default behavior
----------------
Upstream's default tag is derived from the LLVM version::

   LLVM_ABI_NAMESPACE = v${LLVM_VERSION_MAJOR}_${LLVM_VERSION_MINOR}

so a released LLVM 23.0 ships symbols mangled under ``llvm::v23_0``.  This
means:

* Two different LLVM minor versions produce different C++ symbol names.
  Linking or ``dlopen``-loading both into one process produces two
  independent runtime copies of LLVM rather than a crash or silent
  corruption.

* Distributors should further suffix the default (see below) to keep
  their builds distinct from upstream's and from each other.

* Setting ``LLVM_ABI_NAMESPACE=""`` explicitly disables tagging.  The
  expanded code is identical to pre-tag releases; this mode exists for
  consumers that need bit-identical mangling (e.g. for downstream tooling
  that greps by raw symbol, or for strict ABI archaeology).

Configuration
-------------
Relevant CMake variables:

``LLVM_ABI_NAMESPACE``
    Identifier for the inline namespace.  Defaults to
    ``v${LLVM_VERSION_MAJOR}_${LLVM_VERSION_MINOR}``.  Empty disables
    tagging.  Must match the C identifier grammar
    ``^[A-Za-z_][A-Za-z0-9_]*$`` when set; CMake errors out otherwise.

``MLIR_ABI_NAMESPACE``
    Identifier for MLIR's inline namespace.  Defaults to the current
    ``LLVM_ABI_NAMESPACE`` value.  Override independently to retag MLIR
    while leaving LLVM alone (rare, for experiments).

Distributors
~~~~~~~~~~~~
Distributors shipping a compiler that embeds LLVM — in particular any
vendor that plans to coexist with other LLVM consumers in one address
space — are strongly encouraged to suffix the default::

   cmake -DLLVM_ABI_NAMESPACE=v23_0_acme \
         -DMLIR_ABI_NAMESPACE=v23_0_acme ...

This prevents a crash when a user loads (say) Acme's compiler plugin
into another vendor's process that also links LLVM 23.0.

Embedders
~~~~~~~~~
If you embed LLVM as a C++ API consumer, you do not need to change your
code.  ``llvm::Value``, ``mlir::Type``, and every other public type
continue to work exactly as before thanks to ``inline namespace``.  The
only observable effect is in symbol dumps, debuggers, and linker
diagnostics.

C API exclusion
---------------
The LLVM-C and MLIR-C APIs are **not** tagged.  Functions in
``extern "C"`` blocks, ``include/llvm-c/``, ``include/mlir-c/``, and any
translation unit that is part of a C-API library never see
``LLVM_NAMESPACE_BEGIN/END`` wrapping.  This guarantees:

* C clients link against the same ``LLVMCreateContext`` symbol name they
  always did.

* ``dlsym("LLVMCreateContext")`` keeps working across tagged and
  untagged builds.

* A compiler that binds to LLVM via its C API is immune to C++ ABI
  mismatches.

If you need cross-build ABI stability, use the C API.  If you need the
C++ API, link against exactly one copy of LLVM and use its tagged
namespace.

Demangle library
~~~~~~~~~~~~~~~~
``LLVMDemangle`` (``llvm/include/llvm/Demangle`` and
``llvm/lib/Demangle``) is intentionally kept outside the tagged scope:

* It is used by external symbolicators (``perf``, ``llvm-objdump``,
  lldb's embedded plugins, ``libc++abi``) that expect stable
  cross-version symbols for ``llvm::demangle`` / ``llvm::ms_demangle``.

* It is designed to be standalone — it does not link against
  ``LLVMSupport`` and is sometimes vendored directly by consumers.

* Its namespace is controlled by ``DEMANGLE_NAMESPACE_BEGIN`` in
  ``DemangleConfig.h`` rather than by the ABI macros, so mixing the two
  produced mangling mismatches during the initial migration.

The CI backstop skips these directories, and ``DEMANGLE_NAMESPACE_BEGIN``
continues to expand to raw ``namespace llvm { namespace itanium_demangle
{ ... } }``.  Consumers that want symbol-collision isolation on demangle
internals should vendor the library.

Macros
------
Headers that used to spell ``namespace llvm { ... }`` now use macros
defined in ``llvm/Support/ABINamespace.h``:

``LLVM_NAMESPACE_BEGIN`` / ``LLVM_NAMESPACE_END``
    File-scope namespace wrappers.  Replace every raw ``namespace llvm {``
    and matching ``}`` at file scope.

``LLVM_ABI_NS``
    Nested-name-specifier that resolves to ``llvm::vX_Y`` when the tag is
    active and plain ``llvm`` when it is not.  Use it at the start of any
    **out-of-class** definition that would otherwise spell
    ``llvm::Foo`` at file scope, for example::

        struct LLVM_ABI_NS::Foo { int x; };                // was: struct llvm::Foo
        void LLVM_ABI_NS::Foo::bar() {}                    // was: void llvm::Foo::bar()
        template class LLVM_ABI_NS::SmallVectorBase<int>;  // was: template class llvm::...

    This is required because the Itanium C++ ABI mangles such definitions
    by the **spelled** qualifier, not by name lookup.  Without
    ``LLVM_ABI_NS``, a tagged build would mangle the definition as
    ``llvm::Foo`` and fail to resolve against the declaration which
    correctly resides in ``llvm::v23_0::Foo``.

MLIR has the parallel ``MLIR_NAMESPACE_BEGIN``, ``MLIR_NAMESPACE_END``,
and ``MLIR_ABI_NS`` macros defined in ``mlir/Support/ABINamespace.h``.

Authoring guidance
------------------
When adding new LLVM/MLIR C++ code:

1. **Use the macros.** ``LLVM_NAMESPACE_BEGIN`` at the outermost
   namespace opening, ``LLVM_NAMESPACE_END`` at its close.  Keep nested
   namespaces as ordinary ``namespace detail { ... }`` inside.

2. **Do not hand-write ``namespace llvm {`` at file scope.**  The
   rewriter and the CI backstop (``llvm/utils/check-abi-namespace.py``)
   will flag such cases.

3. **Qualify out-of-line definitions with ``LLVM_ABI_NS::``.**  Member
   function definitions, explicit template instantiations, DenseMapInfo
   specializations, ``DOTGraphTraits`` specializations, and similar
   constructs must use the macro-qualified form.

4. **TableGen emitters** generate code through
   ``llvm::NamespaceEmitter`` (``llvm/TableGen/CodeGenHelpers.h``), which
   emits ``LLVM_NAMESPACE_BEGIN/END`` or ``MLIR_NAMESPACE_BEGIN/END``
   whenever the target namespace starts with ``llvm`` or ``mlir``.  New
   emitters should use this helper rather than emitting raw
   ``namespace X {`` strings.

Debugging tagged builds
-----------------------
The tag appears in tool output:

* ``nm -C`` and ``c++filt`` render the demangled name as
  ``llvm::v23_0::Foo``.
* ``gdb`` and ``lldb`` accept both ``llvm::Foo`` and ``llvm::v23_0::Foo``
  in breakpoint commands because inline namespaces participate in name
  lookup.
* Libraries like ``LLVMSupport.a`` contain only tagged symbols for LLVM
  entities (verified by the unit test in
  ``llvm/unittests/Support/ABINamespaceTest.cpp``).

To sanity-check symbol mangling after a build, run::

    nm -g --defined-only $BUILD/lib/libLLVMSupport.a \
      | grep _ZN4llvm | grep -v _ZN4llvm5v23_0

and verify that no C++-mangled (``_ZN4llvm``) symbols are missing the
``5v23_0`` component.  The ABI namespace unit test performs this check
automatically when RTTI is available.

C API preservation check
~~~~~~~~~~~~~~~~~~~~~~~~
C API symbols live in their own namespaces (``LLVMFoo``, ``mlirFoo``)
and must remain untagged.  Verify with::

    nm -g --defined-only $BUILD/lib/libLLVMCore.a | grep '^T LLVM' | head

The output should contain raw ``T LLVMContextCreate`` and friends — not
``T _ZN4llvm5v23_0...``.

Demangler and debugger integration
----------------------------------
The inline versioned namespace is a normal C++ inline namespace from the
point of view of every tool in the toolchain — no special demangler,
pretty-printer, or debugger plugin is required.  This section captures
the behaviour that users and packagers can rely on.

c++filt / llvm-cxxfilt
~~~~~~~~~~~~~~~~~~~~~~
The default Itanium demangler emits the full, tagged name::

    $ echo _ZN4llvm5v23_05ValueC1ENS0_4TypeE | llvm-cxxfilt
    llvm::v23_0::Value::Value(llvm::v23_0::Type)

Use ``--no-params`` to omit argument types for a more compact form.  The
``v23_0`` component always appears on C++-mangled LLVM/MLIR symbols; C
API symbols (``LLVM*`` / ``mlir*``) never do.

If you specifically want to strip the version substring for
grep-compatibility with pre-tag releases, a sed post-filter works::

    llvm-cxxfilt < symbols.txt | sed -E 's/(llvm|mlir)::v[0-9]+_[0-9]+:://g'

This is a purely cosmetic step for human-readable log scraping; do
**not** use it on symbols that are about to be fed into a build, linker,
or ``dlsym`` call.

gdb
~~~
GDB treats an inline namespace exactly as the C++ standard requires:
names are first looked up as if the inline namespace were transparent,
so all of the following are equivalent and all of them work::

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
interpreters are inline-namespace aware as of LLDB 18.  Short-form
names resolve through the inline namespace::

    (lldb) b llvm::Value::setName
    (lldb) expr ((llvm::StringRef) ref).size()

When LLDB is itself *built with* ``LLVM_ABI_NAMESPACE`` active, its
internal ``lldb_private::`` symbols are unaffected (LLDB does not
tag its own namespaces).  Only the embedded LLVM/MLIR libraries carry
the tag.

Profiling / perf
~~~~~~~~~~~~~~~~
``perf report``, FlameGraphs, and similar tools display the tagged
demangled name exactly the way ``c++filt`` does.  When aggregating
profiles across versions, strip the version component with the sed
filter above before bucketing by symbol name; otherwise symbols from
LLVM 23.0 and 23.1 will appear as distinct entries even when they come
from the same source-level function.

Core dumps
~~~~~~~~~~
A core dump captures mangled symbols as they appear in the binary.
Analysing it requires the exact set of LLVM/MLIR libraries used at
crash time, including their tag.  The tag is a *compile-time* preprocessor
define (see ``llvm/Support/ABINamespaceTag.h``), so it is not stamped
as a runtime string by default.  Record the tag in the distribution's
build metadata (``LLVM_ABI_NAMESPACE`` CMake variable) and retrieve it
from any shipped library with ``nm``::

    $ nm -g --defined-only libLLVMSupport.so \
        | grep ' _ZN4llvm' | head -1 | c++filt
    llvm::v23_0::<some symbol>

The ``v23_0`` component of any demangled C++ symbol is the tag — the
build's identity from LLVM's point of view.

Distributors who need a runtime-readable tag are free to add a small
``const char kLLVMABINamespace[] = "v23_0_acme";`` constant derived from
``LLVM_ABI_NAMESPACE`` in their packaging repository; the upstream
tree keeps the tag macro-only to avoid per-TU string constants.

Related design decisions
------------------------
* ``-fvisibility=hidden`` alone does not isolate two LLVM copies; the
  tag complements it.
* The tag is chosen as an inline namespace (rather than a plain outer
  namespace) so ``llvm::`` remains usable source-level; only the mangled
  form changes.
* Header-only templates that consumers specialize for their own types
  keep working because the primary template and its specialization see
  the same ``llvm`` qualifier and the same inline namespace.
* TypeID and ``dyn_cast`` machinery is unaffected: both rely on
  per-translation-unit static storage that is uniqued by tagged class
  identity, not by raw symbol name.

Migration history
-----------------
* Stage 1 (LLVM 23): Introduced macros and the CMake variable.
* Stage 2: Applied ``LLVM_NAMESPACE_BEGIN/END`` tree-wide via
  ``llvm/utils/abi-namespace-wrap/abi_namespace_wrap.py``.
* Stage 3: Validated full build and test suites with the tag active.
* Stage 4: Flipped the CMake default to ``v${MAJOR}_${MINOR}``.
* Stage 5: Documented the tag and added a CI backstop.

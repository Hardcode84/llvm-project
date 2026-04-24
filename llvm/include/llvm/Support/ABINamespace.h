//===- llvm/Support/ABINamespace.h - Inline versioned namespace -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Defines the namespace-wrapping macros that implement LLVM's inline
/// versioned namespace. They embed a build-specific tag into the mangled
/// names of symbols under \c namespace\ llvm so two independently-built
/// copies of the LLVM C++ API can coexist in one process.
///
/// Usage at file scope (the rewriter emits this form):
/// \code
///   LLVM_NAMESPACE_BEGIN
///   // ... declarations or definitions in namespace llvm ...
///   LLVM_NAMESPACE_END
/// \endcode
///
/// Upstream's default tag is derived from the LLVM version — for example
/// \c v23_0 — so a default build expands the macros to:
/// \code
///   namespace llvm { inline namespace v23_0 {
///   // ...
///   } }
/// \endcode
/// and mangled names become \c _ZN4llvm5v23_0... while source-level
/// \c llvm::Value keeps resolving via the \c inline keyword.
///
/// Setting \c -DLLVM_ABI_NAMESPACE="" at CMake time disables tagging; the
/// macros then expand to plain \c namespace\ llvm\ {} and mangled names
/// are identical to pre-tag releases. Never redefine
/// \c LLVM_ABI_NAMESPACE on the consumer compile line — the value must
/// match the installed \c ABINamespaceTag.h sitting next to the installed
/// headers.
///
/// Not every \c namespace\ llvm block is tagged: the Demangle library,
/// the profile/coverage data format libraries, and compiler-rt are
/// intentionally untagged so they can be shared byte-for-byte with
/// standalone runtimes and external tools.
///
/// See llvm/docs/ABIInlineNamespace.rst for the full design, the list of
/// untagged ABI-stable zones, and the distribution/ELF requirements
/// for real dual-LLVM embedding.
//===----------------------------------------------------------------------===//

#ifndef LLVM_SUPPORT_ABINAMESPACE_H
#define LLVM_SUPPORT_ABINAMESPACE_H

#include "llvm/Support/ABINamespaceTag.h"

#ifdef LLVM_ABI_NAMESPACE_ACTIVE
#define LLVM_NAMESPACE_BEGIN                                                   \
  namespace llvm {                                                             \
  inline namespace LLVM_ABI_NAMESPACE {
#define LLVM_NAMESPACE_END                                                     \
  }                                                                            \
  }
/// Nested-name-specifier that resolves to \c llvm::vX_Y when the ABI tag is
/// active and plain \c llvm otherwise. Use this at the *start* of any
/// out-of-class member/struct/enum definition that would otherwise spell
/// \c llvm::Foo\ { … } at file scope. Example:
/// \code
///   struct LLVM_ABI_NS::Foo { int x; };        // was: struct llvm::Foo
///   void LLVM_ABI_NS::Foo::bar() {}            // was: void llvm::Foo::bar()
/// \endcode
/// This is required because Itanium name mangling for out-of-class definitions
/// does not honor the \c inline keyword on the enclosing namespace — the
/// spelled-out qualifier wins, so the definition would be mangled into
/// \c llvm:: rather than \c llvm::vX_Y:: and fail to match its declaration.
///
/// \c LLVM_ABI_NS is unrelated to \c LLVM_ABI from Compiler.h; the latter
/// is a visibility annotation for the public shared-library surface. The
/// two macros share a prefix for historical reasons only.
#define LLVM_ABI_NS llvm::LLVM_ABI_NAMESPACE
#else
#define LLVM_NAMESPACE_BEGIN namespace llvm {
#define LLVM_NAMESPACE_END }
#define LLVM_ABI_NS llvm
#endif

#endif // LLVM_SUPPORT_ABINAMESPACE_H

//===- llvm/Support/ABINamespace.h - Inline versioned namespace -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Defines the namespace-wrapping macros that implement LLVM's optional
/// inline versioned namespace. These let two independently-built copies of
/// LLVM coexist in one process for consumers that link the C++ API, by
/// embedding a build-specific tag into every mangled name under
/// \c namespace llvm.
///
/// Usage at file scope (coordinator rewriter emits this form):
/// \code
///   LLVM_NAMESPACE_BEGIN
///   // ... declarations or definitions in namespace llvm ...
///   LLVM_NAMESPACE_END
/// \endcode
///
/// When the build's tag is empty (the default), the macros expand to plain
/// \c namespace\ llvm\ { and \c }. Mangled names are unchanged.
///
/// When the tag is set (e.g. \c -DLLVM_ABI_NAMESPACE=v23_0), the macros
/// expand to:
/// \code
///   namespace llvm { inline namespace v23_0 {
///   // ...
///   } }
/// \endcode
/// so the mangled name becomes \c _ZN4llvm5v23_0... while source-level
/// \c llvm::Value keeps resolving via the \c inline keyword.
///
/// See llvm/docs/InterfaceExportAnnotations.rst and the migration plan for
/// design, rationale, and the C API exclusion list.
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
#define LLVM_ABI_NS llvm::LLVM_ABI_NAMESPACE
#else
#define LLVM_NAMESPACE_BEGIN namespace llvm {
#define LLVM_NAMESPACE_END }
#define LLVM_ABI_NS llvm
#endif

#endif // LLVM_SUPPORT_ABINAMESPACE_H

//===- mlir/Support/ABINamespace.h - MLIR inline versioned ns --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// MLIR mirror of llvm/Support/ABINamespace.h. See that header for full
/// design rationale. Usage:
/// \code
///   MLIR_NAMESPACE_BEGIN
///   // ... declarations or definitions in namespace mlir ...
///   MLIR_NAMESPACE_END
/// \endcode
///
/// LLVM and MLIR keep distinct tag macros (and therefore distinct inline
/// namespace names) so that an installation can retag MLIR independently
/// of LLVM when that is desired. The default build ties them together by
/// setting `MLIR_ABI_NAMESPACE = ${LLVM_ABI_NAMESPACE}`.
//===----------------------------------------------------------------------===//

#ifndef MLIR_SUPPORT_ABINAMESPACE_H
#define MLIR_SUPPORT_ABINAMESPACE_H

#include "mlir/Support/ABINamespaceTag.h"

#ifdef MLIR_ABI_NAMESPACE_ACTIVE
#define MLIR_NAMESPACE_BEGIN                                                   \
  namespace mlir {                                                             \
  inline namespace MLIR_ABI_NAMESPACE {
#define MLIR_NAMESPACE_END                                                     \
  }                                                                            \
  }
/// Nested-name-specifier that resolves to \c mlir::vX_Y when the ABI tag is
/// active and plain \c mlir otherwise. Use for out-of-class struct/class/enum
/// or member-function definitions that would otherwise spell \c mlir::Foo at
/// file scope. See llvm/Support/ABINamespace.h for the mangling rationale.
#define MLIR_ABI_NS mlir::MLIR_ABI_NAMESPACE
#else
#define MLIR_NAMESPACE_BEGIN namespace mlir {
#define MLIR_NAMESPACE_END }
#define MLIR_ABI_NS mlir
#endif

#endif // MLIR_SUPPORT_ABINAMESPACE_H

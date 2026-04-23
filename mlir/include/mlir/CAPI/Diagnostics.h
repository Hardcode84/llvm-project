//===- IR.h - C API Utils for MLIR Diagnostics ------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_CAPI_DIAGNOSTICS_H
#define MLIR_CAPI_DIAGNOSTICS_H

#include "mlir-c/Diagnostics.h"
#include <cassert>
#include "mlir/Support/ABINamespace.h"

MLIR_NAMESPACE_BEGIN
class Diagnostic;
MLIR_NAMESPACE_END // namespace mlir

inline mlir::Diagnostic &unwrap(MlirDiagnostic diagnostic) {
  assert(diagnostic.ptr && "unexpected null diagnostic");
  return *(static_cast<mlir::Diagnostic *>(diagnostic.ptr));
}

inline MlirDiagnostic wrap(mlir::Diagnostic &diagnostic) {
  return {&diagnostic};
}

#endif // MLIR_CAPI_DIAGNOSTICS_H

//===- Passes.h - OpenMP Pass Construction and Registration -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_OPENMP_TRANSFORMS_PASSES_H
#define MLIR_DIALECT_OPENMP_TRANSFORMS_PASSES_H

#include "mlir/Pass/Pass.h"
#include "mlir/Support/ABINamespace.h"

MLIR_NAMESPACE_BEGIN

namespace omp {

/// Generate the code for registering conversion passes.
#define GEN_PASS_DECL
#define GEN_PASS_REGISTRATION
#include "mlir/Dialect/OpenMP/Transforms/Passes.h.inc"

} // namespace omp
MLIR_NAMESPACE_END // namespace mlir

#endif // MLIR_DIALECT_LLVMIR_TRANSFORMS_PASSES_H

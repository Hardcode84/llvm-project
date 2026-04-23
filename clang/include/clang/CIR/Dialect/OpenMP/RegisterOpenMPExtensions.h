//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CLANG_CIR_DIALECT_OPENMP_REGISTEROPENMPEXTENSIONS_H
#define CLANG_CIR_DIALECT_OPENMP_REGISTEROPENMPEXTENSIONS_H


#include "mlir/Support/ABINamespace.h"
MLIR_NAMESPACE_BEGIN
class DialectRegistry;
MLIR_NAMESPACE_END // namespace mlir

namespace cir::omp {

void registerOpenMPExtensions(mlir::DialectRegistry &registry);

} // namespace cir::omp

#endif // CLANG_CIR_DIALECT_OPENMP_REGISTEROPENMPEXTENSIONS_H

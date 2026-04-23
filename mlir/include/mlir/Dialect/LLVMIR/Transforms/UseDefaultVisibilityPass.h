//===- UseDefaultVisibilityPass.h - Update default visibility -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_LLVMIR_TRANSFORMS_USEDEFAULTVISIBILITYPASS_H
#define MLIR_DIALECT_LLVMIR_TRANSFORMS_USEDEFAULTVISIBILITYPASS_H

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Support/ABINamespace.h"

MLIR_NAMESPACE_BEGIN
class Pass;

namespace LLVM {

#define GEN_PASS_DECL_LLVMUSEDEFAULTVISIBILITYPASS
#include "mlir/Dialect/LLVMIR/Transforms/Passes.h.inc"

} // namespace LLVM
MLIR_NAMESPACE_END // namespace mlir

#endif // MLIR_DIALECT_LLVMIR_TRANSFORMS_USEDEFAULTVISIBILITYPASS_H

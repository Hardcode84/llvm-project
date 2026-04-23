//===- ConvertVectorToLLVMPass.h - Pass to check Vector->LLVM --- --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef MLIR_CONVERSION_VECTORTOLLVM_CONVERTVECTORTOLLVMPASS_H_
#define MLIR_CONVERSION_VECTORTOLLVM_CONVERTVECTORTOLLVMPASS_H_

#include "mlir/Conversion/VectorToLLVM/ConvertVectorToLLVM.h"
#include "mlir/Dialect/Vector/Transforms/VectorTransforms.h"
#include "mlir/Support/ABINamespace.h"

MLIR_NAMESPACE_BEGIN
class Pass;

#define GEN_PASS_DECL_CONVERTVECTORTOLLVMPASS
#include "mlir/Conversion/Passes.h.inc"
MLIR_NAMESPACE_END // namespace mlir
#endif // MLIR_CONVERSION_VECTORTOLLVM_CONVERTVECTORTOLLVMPASS_H_

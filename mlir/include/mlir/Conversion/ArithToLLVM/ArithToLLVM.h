//===- ArithToLLVM.h - Arith to LLVM dialect conversion ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_CONVERSION_ARITHTOLLVM_ARITHTOLLVM_H
#define MLIR_CONVERSION_ARITHTOLLVM_ARITHTOLLVM_H

#include <memory>
#include "mlir/Support/ABINamespace.h"

MLIR_NAMESPACE_BEGIN

class DialectRegistry;
class LLVMTypeConverter;
class RewritePatternSet;
class Pass;

#define GEN_PASS_DECL_ARITHTOLLVMCONVERSIONPASS
#include "mlir/Conversion/Passes.h.inc"

namespace arith {
void populateArithToLLVMConversionPatterns(const LLVMTypeConverter &converter,
                                           RewritePatternSet &patterns);

void registerConvertArithToLLVMInterface(DialectRegistry &registry);
} // namespace arith
MLIR_NAMESPACE_END // namespace mlir

#endif // MLIR_CONVERSION_ARITHTOLLVM_ARITHTOLLVM_H

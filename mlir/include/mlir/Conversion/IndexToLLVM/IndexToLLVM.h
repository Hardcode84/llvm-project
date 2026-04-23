//===- IndexToLLVM.h - Index to LLVM dialect conversion ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_CONVERSION_INDEXTOLLVM_INDEXTOLLVM_H
#define MLIR_CONVERSION_INDEXTOLLVM_INDEXTOLLVM_H

#include <memory>
#include "mlir/Support/ABINamespace.h"

MLIR_NAMESPACE_BEGIN
class DialectRegistry;
class LLVMTypeConverter;
class RewritePatternSet;
class Pass;

#define GEN_PASS_DECL_CONVERTINDEXTOLLVMPASS
#include "mlir/Conversion/Passes.h.inc"

namespace index {
void populateIndexToLLVMConversionPatterns(const LLVMTypeConverter &converter,
                                           RewritePatternSet &patterns);

void registerConvertIndexToLLVMInterface(DialectRegistry &registry);

} // namespace index
MLIR_NAMESPACE_END // namespace mlir

#endif // MLIR_CONVERSION_INDEXTOLLVM_INDEXTOLLVM_H

//===- ComplexToLibm.h - Utils to convert from the complex dialect --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef MLIR_CONVERSION_COMPLEXTOLIBM_COMPLEXTOLIBM_H_
#define MLIR_CONVERSION_COMPLEXTOLIBM_COMPLEXTOLIBM_H_

#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Support/ABINamespace.h"

MLIR_NAMESPACE_BEGIN
template <typename T>
class OperationPass;

#define GEN_PASS_DECL_CONVERTCOMPLEXTOLIBM
#include "mlir/Conversion/Passes.h.inc"

/// Populate the given list with patterns that convert from Complex to Libm
/// calls.
void populateComplexToLibmConversionPatterns(RewritePatternSet &patterns,
                                             PatternBenefit benefit);

MLIR_NAMESPACE_END // namespace mlir

#endif // MLIR_CONVERSION_COMPLEXTOLIBM_COMPLEXTOLIBM_H_

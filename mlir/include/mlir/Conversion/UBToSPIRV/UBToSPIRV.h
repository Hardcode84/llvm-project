//===- UBToSPIRV.h - UB to SPIR-V dialect conversion ------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_CONVERSION_UBTOSPIRV_UBSPIRV_H
#define MLIR_CONVERSION_UBTOSPIRV_UBSPIRV_H

#include <memory>
#include "mlir/Support/ABINamespace.h"

MLIR_NAMESPACE_BEGIN

class SPIRVTypeConverter;
class RewritePatternSet;
class Pass;

#define GEN_PASS_DECL_UBTOSPIRVCONVERSIONPASS
#include "mlir/Conversion/Passes.h.inc"

namespace ub {
void populateUBToSPIRVConversionPatterns(const SPIRVTypeConverter &converter,
                                         RewritePatternSet &patterns);
} // namespace ub
MLIR_NAMESPACE_END // namespace mlir

#endif // MLIR_CONVERSION_UBTOSPIRV_UBSPIRV_H

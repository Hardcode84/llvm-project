//===- MathToAPFloat.h - Math to APFloat impl conversion ---*- C++ ------*-===//
//
// Part of the APFloat Project, under the Apache License v2.0 with APFloat
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH APFloat-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_CONVERSION_ARITHANDMATHTOAPFLOAT_MATHTOAPFLOAT_H
#define MLIR_CONVERSION_ARITHANDMATHTOAPFLOAT_MATHTOAPFLOAT_H

#include <memory>
#include "mlir/Support/ABINamespace.h"

MLIR_NAMESPACE_BEGIN
class Pass;

#define GEN_PASS_DECL_MATHTOAPFLOATCONVERSIONPASS
#include "mlir/Conversion/Passes.h.inc"
MLIR_NAMESPACE_END // namespace mlir

#endif // MLIR_CONVERSION_ARITHANDMATHTOAPFLOAT_MATHTOAPFLOAT_H

//===- Transforms.h - ArmNeon Transformation Entrypoints --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_ARMNEON_TRANSFORMS_H
#define MLIR_DIALECT_ARMNEON_TRANSFORMS_H


#include "mlir/Support/ABINamespace.h"
MLIR_NAMESPACE_BEGIN
class RewritePatternSet;

namespace arm_neon {
void populateLowerContractionToNeonI8MMPatterns(RewritePatternSet &patterns);
void populateLowerContractionToNeonBFMMLAPatterns(RewritePatternSet &patterns);
} // namespace arm_neon

MLIR_NAMESPACE_END // namespace mlir

#endif // MLIR_DIALECT_ARMNEON_TRANSFORMS_H

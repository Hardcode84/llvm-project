//===- MathToFuncs.h - Math to outlined impl conversion ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_CONVERSION_MATHTOFUNCS_MATHTOFUNCS_H
#define MLIR_CONVERSION_MATHTOFUNCS_MATHTOFUNCS_H

#include <memory>
#include "mlir/Support/ABINamespace.h"

MLIR_NAMESPACE_BEGIN
class Pass;

#define GEN_PASS_DECL_CONVERTMATHTOFUNCS
#include "mlir/Conversion/Passes.h.inc"

MLIR_NAMESPACE_END // namespace mlir

#endif // MLIR_CONVERSION_MATHTOFUNCS_MATHTOFUNCS_H

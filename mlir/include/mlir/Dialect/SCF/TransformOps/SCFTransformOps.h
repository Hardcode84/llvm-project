//===- SCFTransformOps.h - SCF transformation ops ---------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_SCF_TRANSFORMOPS_SCFTRANSFORMOPS_H
#define MLIR_DIALECT_SCF_TRANSFORMOPS_SCFTRANSFORMOPS_H

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/Dialect/Transform/IR/TransformTypes.h"
#include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "mlir/Support/ABINamespace.h"

MLIR_NAMESPACE_BEGIN
namespace func {
class FuncOp;
} // namespace func
namespace scf {
class ForallOp;
class ForOp;
class IfOp;
} // namespace scf
MLIR_NAMESPACE_END // namespace mlir

#define GET_OP_CLASSES
#include "mlir/Dialect/SCF/TransformOps/SCFTransformOps.h.inc"

MLIR_NAMESPACE_BEGIN
class DialectRegistry;

namespace scf {
void registerTransformDialectExtension(DialectRegistry &registry);
} // namespace scf
MLIR_NAMESPACE_END // namespace mlir

#endif // MLIR_DIALECT_SCF_TRANSFORMOPS_SCFTRANSFORMOPS_H

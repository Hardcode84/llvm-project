//===- FuncOps.h - Func Dialect Operations ----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_FUNC_IR_OPS_H
#define MLIR_DIALECT_FUNC_IR_OPS_H

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/Support/Compiler.h"
#include "mlir/Support/ABINamespace.h"

MLIR_NAMESPACE_BEGIN
class PatternRewriter;
MLIR_NAMESPACE_END // namespace mlir

#define GET_OP_CLASSES
#include "mlir/Dialect/Func/IR/FuncOps.h.inc"

#include "mlir/Dialect/Func/IR/FuncOpsDialect.h.inc"

LLVM_NAMESPACE_BEGIN

/// Allow stealing the low bits of FuncOp.
template <>
struct PointerLikeTypeTraits<mlir::func::FuncOp> {
  static inline void *getAsVoidPointer(mlir::func::FuncOp val) {
    return const_cast<void *>(val.getAsOpaquePointer());
  }
  static inline mlir::func::FuncOp getFromVoidPointer(void *p) {
    return mlir::func::FuncOp::getFromOpaquePointer(p);
  }
  static constexpr int NumLowBitsAvailable = 3;
};
LLVM_NAMESPACE_END // namespace llvm

#endif // MLIR_DIALECT_FUNC_IR_OPS_H

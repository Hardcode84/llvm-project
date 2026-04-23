//===- Async.h - MLIR Async dialect -----------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the async dialect that is used for modeling asynchronous
// execution.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_ASYNC_IR_ASYNC_H
#define MLIR_DIALECT_ASYNC_IR_ASYNC_H

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/Dialect/Async/IR/AsyncTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

//===----------------------------------------------------------------------===//
// Async Dialect
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Async/IR/AsyncOpsDialect.h.inc"

//===----------------------------------------------------------------------===//
// Async Dialect Operations
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "mlir/Dialect/Async/IR/AsyncOps.h.inc"
#include "llvm/Support/Compiler.h"
#include "mlir/Support/ABINamespace.h"

//===----------------------------------------------------------------------===//
// Helper functions of Async dialect transformations.
//===----------------------------------------------------------------------===//

MLIR_NAMESPACE_BEGIN
namespace async {

/// Returns true if the type is reference counted at runtime.
inline bool isRefCounted(Type type) {
  return isa<TokenType, ValueType, GroupType>(type);
}

} // namespace async
MLIR_NAMESPACE_END // namespace mlir

LLVM_NAMESPACE_BEGIN

/// Allow stealing the low bits of async::FuncOp.
template <>
struct PointerLikeTypeTraits<mlir::async::FuncOp> {
  static inline void *getAsVoidPointer(mlir::async::FuncOp val) {
    return const_cast<void *>(val.getAsOpaquePointer());
  }
  static inline mlir::async::FuncOp getFromVoidPointer(void *p) {
    return mlir::async::FuncOp::getFromOpaquePointer(p);
  }
  static constexpr int NumLowBitsAvailable = 3;
};
LLVM_NAMESPACE_END // namespace llvm

#endif // MLIR_DIALECT_ASYNC_IR_ASYNC_H

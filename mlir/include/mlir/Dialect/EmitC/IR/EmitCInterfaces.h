//===- EmitCInterfaces.h - EmitC interfaces definitions ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares C++ classes for some of the interfaces used in the EmitC
// dialect.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_EMITC_IR_EMITCINTERFACES_H
#define MLIR_DIALECT_EMITC_IR_EMITCINTERFACES_H

#include "mlir/IR/OpDefinition.h"
#include "mlir/Support/ABINamespace.h"

MLIR_NAMESPACE_BEGIN
namespace emitc {
//
} // namespace emitc
MLIR_NAMESPACE_END // namespace mlir

//===----------------------------------------------------------------------===//
// EmitC Dialect Interfaces
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/EmitC/IR/EmitCInterfaces.h.inc"

#endif // MLIR_DIALECT_EMITC_IR_EMITCINTERFACES_H

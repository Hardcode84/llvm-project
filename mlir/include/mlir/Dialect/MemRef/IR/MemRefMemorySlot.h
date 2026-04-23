//===- MemRefMemorySlot.h - Implementation of Memory Slot Interfaces ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_MEMREF_IR_MEMREFMEMORYSLOT_H
#define MLIR_DIALECT_MEMREF_IR_MEMREFMEMORYSLOT_H


#include "mlir/Support/ABINamespace.h"
MLIR_NAMESPACE_BEGIN
class DialectRegistry;

namespace memref {
void registerMemorySlotExternalModels(DialectRegistry &registry);
} // namespace memref
MLIR_NAMESPACE_END // namespace mlir

#endif // MLIR_DIALECT_MEMREF_IR_MEMREFMEMORYSLOT_H

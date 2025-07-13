//===- MemoryAccessOpInterfaces.cpp ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Interfaces/MemoryAccessOpInterfaces.h"
// #include "mlir/IR/Operation.h"
// #include "mlir/IR/Value.h"

//===----------------------------------------------------------------------===//
// IndexedAccessOpInterface and IndexedMemCpyOpInterface
//===----------------------------------------------------------------------===//

namespace mlir::memref {

#include "mlir/Interfaces/MemoryAccessOpInterfaces.cpp.inc"

} // namespace mlir::memref

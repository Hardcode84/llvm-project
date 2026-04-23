//===- FoldInterfaces.h - Folding Interfaces --------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef MLIR_INTERFACES_FOLDINTERFACES_H_
#define MLIR_INTERFACES_FOLDINTERFACES_H_

#include "mlir/IR/DialectInterface.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Support/ABINamespace.h"

MLIR_NAMESPACE_BEGIN
class Attribute;
class OpFoldResult;
class Region;
MLIR_NAMESPACE_END // namespace mlir

#include "mlir/Interfaces/DialectFoldInterface.h.inc"

#endif // MLIR_INTERFACES_FOLDINTERFACES_H_

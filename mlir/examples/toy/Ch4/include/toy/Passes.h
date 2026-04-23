//===- Passes.h - Toy Passes Definition -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file exposes the entry points to create compiler passes for Toy.
//
//===----------------------------------------------------------------------===//

#ifndef TOY_PASSES_H
#define TOY_PASSES_H

#include <memory>
#include "mlir/Support/ABINamespace.h"

MLIR_NAMESPACE_BEGIN
class Pass;

namespace toy {
std::unique_ptr<Pass> createShapeInferencePass();
} // namespace toy
MLIR_NAMESPACE_END // namespace mlir

#endif // TOY_PASSES_H

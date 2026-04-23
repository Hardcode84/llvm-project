//===- RuntimeOpVerification.h - Op Verification ----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_TENSOR_RUNTIMEOPVERIFICATION_H
#define MLIR_DIALECT_TENSOR_RUNTIMEOPVERIFICATION_H


#include "mlir/Support/ABINamespace.h"
MLIR_NAMESPACE_BEGIN
class DialectRegistry;

namespace tensor {
void registerRuntimeVerifiableOpInterfaceExternalModels(
    DialectRegistry &registry);
} // namespace tensor
MLIR_NAMESPACE_END // namespace mlir

#endif // MLIR_DIALECT_TENSOR_RUNTIMEOPVERIFICATION_H

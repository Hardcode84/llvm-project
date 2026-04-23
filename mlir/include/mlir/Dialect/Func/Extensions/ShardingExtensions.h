//===- ShardingExtensions.h - -----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_FUNC_IR_SHARDINGINTERFACEIMPL_H_
#define MLIR_DIALECT_FUNC_IR_SHARDINGINTERFACEIMPL_H_


#include "mlir/Support/ABINamespace.h"
MLIR_NAMESPACE_BEGIN

class DialectRegistry;

namespace func {

void registerShardingInterfaceExternalModels(DialectRegistry &registry);

} // namespace func
MLIR_NAMESPACE_END // namespace mlir

#endif // MLIR_DIALECT_FUNC_IR_SHARDINGINTERFACEIMPL_H_

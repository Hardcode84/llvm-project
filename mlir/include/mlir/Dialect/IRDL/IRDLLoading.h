//===- IRDLRegistration.h - IRDL registration -------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Manages the registration of MLIR objects from IRDL operations.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_IRDL_IRDLREGISTRATION_H
#define MLIR_DIALECT_IRDL_IRDLREGISTRATION_H


#include "llvm/Support/Compiler.h"
#include "mlir/Support/ABINamespace.h"
LLVM_NAMESPACE_BEGIN
struct LogicalResult;
LLVM_NAMESPACE_END // namespace llvm

MLIR_NAMESPACE_BEGIN
class ModuleOp;
MLIR_NAMESPACE_END // namespace mlir

MLIR_NAMESPACE_BEGIN
namespace irdl {

/// Load all the dialects defined in the module.
llvm::LogicalResult loadDialects(ModuleOp op);

} // namespace irdl
MLIR_NAMESPACE_END // namespace mlir

#endif // MLIR_DIALECT_IRDL_IRDLREGISTRATION_H

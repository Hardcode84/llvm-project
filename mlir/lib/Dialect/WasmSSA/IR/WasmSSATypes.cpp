//===- WasmSSAOps.cpp - WasmSSA dialect operations ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===---------------------------------------------------------------------===//

#include "mlir/Dialect/WasmSSA/IR/WasmSSA.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Types.h"
#include "llvm/Support/LogicalResult.h"

#include <optional>
#include "mlir/Support/ABINamespace.h"

MLIR_NAMESPACE_BEGIN
namespace wasmssa {
#include "mlir/Dialect/WasmSSA/IR/WasmSSATypeConstraints.cpp.inc"
}
MLIR_NAMESPACE_END // namespace mlir::wasmssa

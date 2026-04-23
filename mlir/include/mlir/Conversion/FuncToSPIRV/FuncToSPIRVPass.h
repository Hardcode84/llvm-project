//===- FuncToSPIRVPass.h - Func to SPIR-V Passes ----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Provides passes to convert Func dialect to SPIR-V dialect.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_CONVERSION_FUNCTOSPIRV_FUNCTOSPIRVPASS_H
#define MLIR_CONVERSION_FUNCTOSPIRV_FUNCTOSPIRVPASS_H

#include "mlir/Pass/Pass.h"
#include "mlir/Support/ABINamespace.h"

MLIR_NAMESPACE_BEGIN
class ModuleOp;

#define GEN_PASS_DECL_CONVERTFUNCTOSPIRVPASS
#include "mlir/Conversion/Passes.h.inc"

MLIR_NAMESPACE_END // namespace mlir

#endif // MLIR_CONVERSION_FUNCTOSPIRV_FUNCTOSPIRVPASS_H

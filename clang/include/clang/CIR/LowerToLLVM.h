//====- LowerToLLVM.h- Lowering from CIR to LLVM --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares an interface for converting CIR modules to LLVM IR.
//
//===----------------------------------------------------------------------===//
#ifndef CLANG_CIR_LOWERTOLLVM_H
#define CLANG_CIR_LOWERTOLLVM_H

#include "llvm/ADT/StringRef.h"
#include <memory>
#include "llvm/Support/Compiler.h"
#include "mlir/Support/ABINamespace.h"

LLVM_NAMESPACE_BEGIN
class LLVMContext;
class Module;
LLVM_NAMESPACE_END // namespace llvm

MLIR_NAMESPACE_BEGIN
class ModuleOp;
MLIR_NAMESPACE_END // namespace mlir

namespace cir {

namespace direct {
std::unique_ptr<llvm::Module>
lowerDirectlyFromCIRToLLVMIR(mlir::ModuleOp mlirModule,
                             llvm::LLVMContext &llvmCtx,
                             llvm::StringRef mlirSaveTempsOutFile = {});
} // namespace direct
} // namespace cir

#endif // CLANG_CIR_LOWERTOLLVM_H

//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares an interface for running CIR-to-CIR passes.
//
//===----------------------------------------------------------------------===//

#ifndef CLANG_CIR_CIRTOCIRPASSES_H
#define CLANG_CIR_CIRTOCIRPASSES_H

#include "mlir/Pass/Pass.h"

#include <memory>
#include "mlir/Support/ABINamespace.h"

namespace clang {
class ASTContext;
}

MLIR_NAMESPACE_BEGIN
class MLIRContext;
class ModuleOp;
MLIR_NAMESPACE_END // namespace mlir

namespace cir {

// Run set of cleanup/prepare/etc passes CIR <-> CIR.
mlir::LogicalResult
runCIRToCIRPasses(mlir::ModuleOp theModule, mlir::MLIRContext &mlirCtx,
                  clang::ASTContext &astCtx, bool enableVerifier,
                  bool enableIdiomRecognizer, bool enableCIRSimplify);

} // namespace cir

#endif // CLANG_CIR_CIRTOCIRPASSES_H_

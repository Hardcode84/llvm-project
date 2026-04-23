//===- PrintIR.cpp - Pass to dump IR on debug stream ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/Support/Debug.h"
#include "mlir/Support/ABINamespace.h"

MLIR_NAMESPACE_BEGIN
#define GEN_PASS_DEF_PRINTIRPASS
#include "mlir/Transforms/Passes.h.inc"
MLIR_NAMESPACE_END // namespace mlir

MLIR_NAMESPACE_BEGIN
namespace {

struct PrintIRPass : public impl::PrintIRPassBase<PrintIRPass> {
  using impl::PrintIRPassBase<PrintIRPass>::PrintIRPassBase;

  void runOnOperation() override {
    llvm::dbgs() << "// -----// IR Dump";
    if (!this->label.empty())
      llvm::dbgs() << " " << this->label;
    llvm::dbgs() << " //----- //\n";
    getOperation()->dump();
    markAllAnalysesPreserved();
  }
};

} // namespace

MLIR_NAMESPACE_END // namespace mlir

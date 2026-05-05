//===- WaveAMDMetadata.cpp - WaveAMD metadata -------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Wave/Transforms/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/WaveMachine/IR/WaveMachine.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"

namespace mlir::wave {
#define GEN_PASS_DEF_WAVEAMDMETADATA
#include "mlir/Dialect/Wave/Transforms/Passes.h.inc"
} // namespace mlir::wave

using namespace mlir;

namespace {

struct WaveAMDMetadataPass
    : public wave::impl::WaveAMDMetadataBase<WaveAMDMetadataPass> {
  void runOnOperation() override {
    if (!getOperation()->hasAttr("wavemachine.target")) {
      getOperation().emitError("waveamd-metadata requires a wavemachine.target "
                               "module attribute");
      return signalPassFailure();
    }
    OpBuilder builder(getOperation().getContext());
    for (func::FuncOp func : getOperation().getOps<func::FuncOp>()) {
      if (!func->hasAttr("wave.kernel"))
        continue;
      if (!func->hasAttr("wavemachine.kernarg_size") ||
          !func->hasAttr("wavemachine.sgpr_count") ||
          !func->hasAttr("wavemachine.vgpr_count")) {
        func.emitError("waveamd-metadata requires ABI and resource attributes "
                       "on kernels");
        return signalPassFailure();
      }
      func->setAttr("wavemachine.metadata", builder.getUnitAttr());
    }
  }
};

} // namespace

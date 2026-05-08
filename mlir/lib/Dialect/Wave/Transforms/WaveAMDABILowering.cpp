//===- WaveAMDABILowering.cpp - WaveAMD ABI lowering ------------*- C++ -*-===//
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
#include "llvm/ADT/STLExtras.h"

namespace mlir::wave {
#define GEN_PASS_DEF_WAVEAMDABILOWERING
#include "mlir/Dialect/Wave/Transforms/Passes.h.inc"
} // namespace mlir::wave

using namespace mlir;

namespace {

static wavemachine::ImmType getImmType(MLIRContext *ctx) {
  return wavemachine::ImmType::get(ctx);
}

static Operation *createWMOp(OpBuilder &builder, Location loc, StringRef name,
                             ValueRange operands, TypeRange resultTypes,
                             ArrayRef<NamedAttribute> attrs = {}) {
  OperationState state(loc, ("wavemachine." + name).str());
  state.addOperands(operands);
  state.addTypes(resultTypes);
  state.addAttributes(attrs);
  return builder.create(state);
}

static Value createImm(OpBuilder &builder, Location loc, int64_t value) {
  Operation *op =
      createWMOp(builder, loc, "imm", {}, getImmType(builder.getContext()),
                 {builder.getNamedAttr("value", builder.getI64IntegerAttr(value))});
  return op->getResult(0);
}

static Value createInstr(OpBuilder &builder, Location loc, StringRef name,
                         ValueRange operands, Type resultType,
                         ArrayRef<NamedAttribute> attrs = {}) {
  Operation *op = createWMOp(builder, loc, name, operands, resultType, attrs);
  return op->getResult(0);
}

static bool isSGPR(wavemachine::RegType type) {
  return type.getRegClass() == wavemachine::RegClass::SGPR;
}

struct WaveAMDABILoweringPass
    : public wave::impl::WaveAMDABILoweringBase<WaveAMDABILoweringPass> {
  void runOnOperation() override {
    ModuleOp module = getOperation();
    OpBuilder builder(module.getContext());
    for (func::FuncOp func : module.getOps<func::FuncOp>()) {
      if (!func->hasAttr("wave.kernel"))
        continue;
      Block &block = func.getBody().front();

      unsigned offset = 0;
      for (Operation &op : llvm::make_early_inc_range(block)) {
        if (!isa<wavemachine::ArgOp>(op))
          continue;
        if (op.getNumResults() != 1) {
          op.emitError("waveamd-abi-lowering expects wavemachine.arg "
                       "to have one result");
          return signalPassFailure();
        }
        auto regType = dyn_cast<wavemachine::RegType>(op.getResult(0).getType());
        if (!regType || !isSGPR(regType)) {
          op.emitError("waveamd-abi-lowering expects kernel arguments "
                       "to be SGPR WaveMachine registers");
          return signalPassFailure();
        }
        auto pointerAttr = op.getAttrOfType<BoolAttr>("pointer");
        if (!pointerAttr) {
          op.emitError("waveamd-abi-lowering expects wavemachine.arg "
                       "to have a pointer attribute");
          return signalPassFailure();
        }
        bool isPointer = pointerAttr.getValue();
        if ((isPointer && regType.getWidth() != 2 && regType.getWidth() != 4) ||
            (!isPointer && regType.getWidth() != 1)) {
          op.emitError("waveamd-abi-lowering found argument register width "
                       "inconsistent with pointer attribute");
          return signalPassFailure();
        }

        builder.setInsertionPoint(&op);
        Value offsetImm = createImm(builder, op.getLoc(), offset);
        StringRef opcode = "s_load_b32";
        if (isPointer)
          opcode = regType.getWidth() == 4 ? "s_load_b128" : "s_load_b64";
        Value loaded = createInstr(
            builder, op.getLoc(), opcode, offsetImm, op.getResult(0).getType(),
            {builder.getNamedAttr("base", builder.getStringAttr("s[0:1]"))});
        op.getResult(0).replaceAllUsesWith(loaded);
        op.erase();
        offset += isPointer ? regType.getWidth() * 4 : 4;
      }
      unsigned kernargSize = (std::max(offset, 4u) + 7u) & ~7u;
      func->setAttr("wavemachine.kernarg_size",
                    builder.getI64IntegerAttr(kernargSize));
    }
  }
};

} // namespace

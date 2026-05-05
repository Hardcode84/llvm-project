//===- WaveAMDHazardWaits.cpp - WaveAMD hazard waits ------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Wave/Transforms/Passes.h"

#include "Utils/AMDGPUBaseInfo.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/WaveMachine/IR/WaveMachine.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/TargetParser/TargetParser.h"
#include <optional>

namespace mlir::wave {
#define GEN_PASS_DEF_WAVEAMDHAZARDWAITS
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

static Operation *createInstrNoResult(OpBuilder &builder, Location loc,
                                      StringRef name, ValueRange operands) {
  return createWMOp(builder, loc, name, operands, TypeRange{});
}

static FailureOr<llvm::AMDGPU::IsaVersion> getIsaVersion(Operation *op) {
  auto module = dyn_cast<ModuleOp>(op);
  if (!module)
    module = op->getParentOfType<ModuleOp>();
  if (!module)
    return op->emitError("waveamd-insert-hazard-waits requires a module");
  auto target = module->getAttrOfType<StringAttr>("wavemachine.target");
  if (!target)
    return module.emitError("waveamd-insert-hazard-waits requires a "
                            "wavemachine.target attribute");
  StringRef cpu = target.getValue();
  std::pair<StringRef, StringRef> split = cpu.rsplit("--");
  if (!split.second.empty())
    cpu = split.second;
  llvm::AMDGPU::IsaVersion version = llvm::AMDGPU::getIsaVersion(cpu);
  if (version.Major == 0)
    return module.emitError("unsupported AMDGPU target: ") << target.getValue();
  return version;
}

static std::optional<unsigned> getImmediate(Value value) {
  Operation *def = value.getDefiningOp();
  if (!def || !isa<wavemachine::ImmOp>(def))
    return std::nullopt;
  return static_cast<unsigned>(
      def->getAttrOfType<IntegerAttr>("value").getInt());
}

struct WaveAMDHazardWaitsPass
    : public wave::impl::WaveAMDHazardWaitsBase<WaveAMDHazardWaitsPass> {
  void runOnOperation() override {
    ModuleOp module = getOperation();
    OpBuilder builder(module.getContext());
    FailureOr<llvm::AMDGPU::IsaVersion> isaVersion = getIsaVersion(module);
    if (failed(isaVersion))
      return signalPassFailure();
    unsigned defaultLgkmcnt =
        llvm::AMDGPU::decodeLgkmcnt(*isaVersion,
                                    llvm::AMDGPU::getWaitcntBitMask(*isaVersion));
    for (func::FuncOp func : module.getOps<func::FuncOp>()) {
      bool pendingLgkmWait = false;
      for (Operation &op : llvm::make_early_inc_range(func.getBody().front())) {
        if (op.getName().getDialectNamespace() !=
            wavemachine::WaveMachineDialect::getDialectNamespace())
          continue;
        if (func->hasAttr("wave.kernel") && isa<wavemachine::ArgOp>(op)) {
          op.emitError("waveamd-insert-hazard-waits expects ABI-lowered kernel "
                       "arguments");
          return signalPassFailure();
        }
        if (op.hasTrait<OpTrait::wavemachine::SMEMLoadOp>() &&
            !op.getAttrOfType<StringAttr>("base")) {
          op.emitError("waveamd-insert-hazard-waits expects scalar memory "
                       "loads to carry a base register attribute");
          return signalPassFailure();
        }

        if (op.hasTrait<OpTrait::wavemachine::VALUOp>() && pendingLgkmWait) {
          builder.setInsertionPoint(&op);
          createInstrNoResult(builder, op.getLoc(), "s_delay_alu",
                              createImm(builder, op.getLoc(), 1));
          pendingLgkmWait = false;
        }

        if (isa<wavemachine::SWaitcntOp>(op)) {
          auto imm = getImmediate(op.getOperand(0));
          if (!imm)
            continue;
          unsigned vm = 0;
          unsigned exp = 0;
          unsigned lg = 0;
          llvm::AMDGPU::decodeWaitcnt(*isaVersion, *imm, vm, exp, lg);
          pendingLgkmWait = lg != defaultLgkmcnt;
        }
      }
    }
  }
};

} // namespace

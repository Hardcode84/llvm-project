//===- WaveAMDRegAlloc.cpp - WaveAMD register allocation --------*- C++ -*-===//
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
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/TargetParser/TargetParser.h"
#include "llvm/TargetParser/Triple.h"
#include <limits>
#include <optional>

namespace mlir::wave {
#define GEN_PASS_DEF_WAVEAMDREGALLOC
#include "mlir/Dialect/Wave/Transforms/Passes.h.inc"
} // namespace mlir::wave

using namespace mlir;

namespace {

struct LiveInterval {
  Operation *def = nullptr;
  unsigned start = std::numeric_limits<unsigned>::max();
  unsigned end = 0;
};

struct RegisterLimits {
  unsigned numSGPR = 0;
  unsigned numVGPR = 0;
};

static bool isReg(Value value) {
  return isa<wavemachine::RegType>(value.getType());
}

static bool isSGPR(wavemachine::RegType type) {
  return type.getRegClass() == wavemachine::RegClass::SGPR;
}

static bool isVGPR(wavemachine::RegType type) {
  return type.getRegClass() == wavemachine::RegClass::VGPR;
}

static FailureOr<std::unique_ptr<llvm::MCSubtargetInfo>>
createSubtargetInfo(ModuleOp module) {
  auto targetAttr = module->getAttrOfType<StringAttr>("wavemachine.target");
  if (!targetAttr)
    return module.emitError("waveamd-reg-alloc requires a wavemachine.target "
                            "attribute");

  StringRef target = targetAttr.getValue();
  std::pair<StringRef, StringRef> split = target.rsplit("--");
  StringRef cpu = split.second.empty() ? target : split.second;

  static llvm::once_flag initializeBackendOnce;
  llvm::call_once(initializeBackendOnce, []() {
    llvm::InitializeAllTargetInfos();
    llvm::InitializeAllTargetMCs();
  });

  llvm::Triple triple("amdgcn-amd-amdhsa");
  std::string error;
  const llvm::Target *llvmTarget =
      llvm::TargetRegistry::lookupTarget(triple, error);
  if (!llvmTarget)
    return module.emitError("failed to lookup AMDGPU target: ") << error;

  std::unique_ptr<llvm::MCSubtargetInfo> sti(
      llvmTarget->createMCSubtargetInfo(triple, cpu, /*Features=*/""));
  if (!sti)
    return module.emitError("unsupported AMDGPU target: ") << target;
  if (llvm::AMDGPU::getIsaVersion(cpu).Major == 0)
    return module.emitError("unsupported AMDGPU target: ") << target;
  return sti;
}

static FailureOr<RegisterLimits> getRegisterLimits(ModuleOp module) {
  FailureOr<std::unique_ptr<llvm::MCSubtargetInfo>> sti =
      createSubtargetInfo(module);
  if (failed(sti))
    return failure();

  RegisterLimits limits;
  limits.numSGPR = llvm::AMDGPU::IsaInfo::getAddressableNumSGPRs(sti->get());
  limits.numVGPR =
      llvm::AMDGPU::IsaInfo::getAddressableNumVGPRs(sti->get(),
                                                    /*DynamicVGPRBlockSize=*/0);
  return limits;
}

struct WaveAMDRegAllocPass
    : public wave::impl::WaveAMDRegAllocBase<WaveAMDRegAllocPass> {
  void runOnOperation() override {
    FailureOr<RegisterLimits> limits = getRegisterLimits(getOperation());
    if (failed(limits))
      return signalPassFailure();
    for (func::FuncOp func : getOperation().getOps<func::FuncOp>()) {
      if (failed(allocateFunction(func, *limits)))
        return signalPassFailure();
    }
  }

  LogicalResult allocateFunction(func::FuncOp func, RegisterLimits limits) {
    SmallVector<Operation *> orderedOps;
    DenseMap<Operation *, unsigned> positions;
    for (Operation &op : func.getBody().front()) {
      positions[&op] = orderedOps.size();
      orderedOps.push_back(&op);
    }

    SmallVector<LiveInterval> sgprs;
    SmallVector<LiveInterval> vgprs;
    DenseMap<Value, unsigned> sgprIntervals;
    DenseMap<Value, unsigned> vgprIntervals;
    for (Operation *op : orderedOps) {
      for (Value result : op->getResults()) {
        if (!isReg(result))
          continue;
        auto regType = cast<wavemachine::RegType>(result.getType());
        if (!isSGPR(regType) && !isVGPR(regType))
          return op->emitError("waveamd-reg-alloc supports only SGPR "
                               "and VGPR register classes");
        if (regType.getIndex() >= 0)
          continue;
        SmallVector<LiveInterval> &bucket =
            isSGPR(regType) ? sgprs : vgprs;
        unsigned index = bucket.size();
        bucket.push_back(LiveInterval{op, positions[op], positions[op]});
        if (isSGPR(regType))
          sgprIntervals[result] = index;
        else
          vgprIntervals[result] = index;
      }
    }

    for (Operation *op : orderedOps) {
      unsigned pos = positions[op];
      for (Value operand : op->getOperands()) {
        if (auto it = sgprIntervals.find(operand); it != sgprIntervals.end())
          sgprs[it->second].end = std::max(sgprs[it->second].end, pos);
        if (auto it = vgprIntervals.find(operand); it != vgprIntervals.end())
          vgprs[it->second].end = std::max(vgprs[it->second].end, pos);
      }
    }

    if (failed(allocateClass(func, sgprs, limits.numSGPR,
                             func->hasAttr("wave.kernel") ? 2 : 0)))
      return failure();
    if (failed(allocateClass(func, vgprs, limits.numVGPR, /*reserved=*/0)))
      return failure();
    return success();
  }

  LogicalResult allocateClass(func::FuncOp func,
                              MutableArrayRef<LiveInterval> intervals,
                              unsigned numPhys, unsigned reserved) {
    llvm::stable_sort(intervals, [](const LiveInterval &lhs,
                                   const LiveInterval &rhs) {
      if (lhs.start != rhs.start)
        return lhs.start < rhs.start;
      return lhs.def->isBeforeInBlock(rhs.def);
    });

    SmallVector<LiveInterval> active;
    SmallVector<bool> used(numPhys, false);
    for (unsigned i = 0; i != reserved && i != numPhys; ++i)
      used[i] = true;

    auto expireOld = [&](unsigned pos) {
      SmallVector<LiveInterval> stillActive;
      for (LiveInterval interval : active) {
        if (interval.end < pos) {
          auto regType =
              cast<wavemachine::RegType>(interval.def->getResult(0).getType());
          unsigned phys = regType.getIndex();
          unsigned width = regType.getWidth();
          for (unsigned i = 0; i != width; ++i)
            used[phys + i] = false;
        } else {
          stillActive.push_back(interval);
        }
      }
      active = std::move(stillActive);
    };

    for (LiveInterval interval : intervals) {
      expireOld(interval.start);
      unsigned width =
          cast<wavemachine::RegType>(interval.def->getResult(0).getType()).getWidth();
      std::optional<unsigned> phys =
          findFreeContiguous(used, width, /*align=*/width);
      if (!phys)
        return func.emitError("WaveMachine register allocator ran out of registers");
      auto oldType =
          cast<wavemachine::RegType>(interval.def->getResult(0).getType());
      interval.def->getResult(0).setType(wavemachine::RegType::get(
          func.getContext(), oldType.getRegClass(), oldType.getWidth(),
          static_cast<int64_t>(*phys)));
      for (unsigned i = 0; i != width; ++i)
        used[*phys + i] = true;
      active.push_back(interval);
      llvm::sort(active, [](const LiveInterval &lhs, const LiveInterval &rhs) {
        return lhs.end < rhs.end;
      });
    }
    return success();
  }

  static std::optional<unsigned> findFreeContiguous(ArrayRef<bool> used,
                                                    unsigned width,
                                                    unsigned align) {
    for (unsigned i = 0, e = used.size(); i + width <= e; ++i) {
      if (i % align)
        continue;
      bool allFree = true;
      for (unsigned j = 0; j != width; ++j) {
        if (used[i + j]) {
          allFree = false;
          break;
        }
      }
      if (allFree)
        return i;
    }
    return std::nullopt;
  }
};

} // namespace

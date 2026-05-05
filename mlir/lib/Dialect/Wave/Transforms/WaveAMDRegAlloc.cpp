//===- WaveAMDRegAlloc.cpp - WaveAMD register allocation --------*- C++ -*-===//
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
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
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

static bool isReg(Value value) {
  return isa<wavemachine::RegType>(value.getType());
}

static bool isSGPR(wavemachine::RegType type) { return type.getRegClass() == 0; }

static bool isVGPR(wavemachine::RegType type) { return type.getRegClass() == 1; }

struct WaveAMDRegAllocPass
    : public wave::impl::WaveAMDRegAllocBase<WaveAMDRegAllocPass> {
  void runOnOperation() override {
    for (func::FuncOp func : getOperation().getOps<func::FuncOp>()) {
      if (failed(allocateFunction(func)))
        return signalPassFailure();
    }
  }

  LogicalResult allocateFunction(func::FuncOp func) {
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
          return op->emitError("waveamd-reg-alloc supports only SGPR(0) "
                               "and VGPR(1) register classes");
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

    if (failed(allocateClass(func, sgprs, /*numPhys=*/32,
                             func->hasAttr("wave.kernel") ? 2 : 0)))
      return failure();
    if (failed(allocateClass(func, vgprs, /*numPhys=*/32, /*reserved=*/0)))
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
          unsigned phys = interval.def->getAttrOfType<IntegerAttr>("phys").getInt();
          unsigned width =
              cast<wavemachine::RegType>(interval.def->getResult(0).getType()).getWidth();
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
      std::optional<unsigned> phys = findFreeContiguous(used, width);
      if (!phys)
        return func.emitError("WaveMachine register allocator ran out of registers");
      interval.def->setAttr("phys", IntegerAttr::get(IntegerType::get(func.getContext(), 64), *phys));
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
                                                    unsigned width) {
    for (unsigned i = 0, e = used.size(); i + width <= e; ++i) {
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

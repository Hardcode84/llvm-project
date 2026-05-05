//===- WaveMachineWaitcnt.cpp - WaveMachine waitcnt insertion ----*- C++ -*-===//
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
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"
#include "llvm/TargetParser/TargetParser.h"
#include <optional>

namespace mlir::wave {
#define GEN_PASS_DEF_WAVEMACHINETICKETWAITS
#include "mlir/Dialect/Wave/Transforms/Passes.h.inc"
} // namespace mlir::wave

using namespace mlir;
using namespace mlir::wave;

namespace {

enum class CounterKind { Vmem, Lgkm, Vscnt };

struct Ticket {
  CounterKind counter;
  int64_t value = -1;
};

struct CounterState {
  int64_t lastTicket = -1;
  std::optional<int64_t> lastWait;

  bool operator==(const CounterState &rhs) const {
    return lastTicket == rhs.lastTicket && lastWait == rhs.lastWait;
  }

  bool operator!=(const CounterState &rhs) const { return !(*this == rhs); }

  void observeIssue(int64_t ticket) {
    if (ticket <= lastTicket)
      return;
    if (lastWait)
      *lastWait += ticket - lastTicket;
    lastTicket = ticket;
  }

  std::optional<unsigned> computeWait(int64_t requiredTicket) const {
    if (requiredTicket < 0 || lastTicket < 0)
      return std::nullopt;
    int64_t threshold = std::max<int64_t>(0, lastTicket - requiredTicket);
    if (lastWait && *lastWait <= threshold)
      return std::nullopt;
    return static_cast<unsigned>(threshold);
  }

  void observeWait(unsigned threshold) {
    if (!lastWait || threshold < *lastWait)
      lastWait = threshold;
  }

  bool merge(const CounterState &other) {
    CounterState old = *this;
    lastTicket = std::max(lastTicket, other.lastTicket);
    if (other.lastWait && (!lastWait || *other.lastWait < *lastWait))
      lastWait = other.lastWait;
    return *this != old;
  }
};

struct WaitRequirement {
  std::optional<unsigned> vmcnt;
  std::optional<unsigned> lgkmcnt;
  std::optional<unsigned> vscnt;

  bool hasWait() const { return vmcnt || lgkmcnt || vscnt; }

  void add(CounterKind counter, unsigned threshold) {
    std::optional<unsigned> *slot = nullptr;
    switch (counter) {
    case CounterKind::Vmem:
      slot = &vmcnt;
      break;
    case CounterKind::Lgkm:
      slot = &lgkmcnt;
      break;
    case CounterKind::Vscnt:
      slot = &vscnt;
      break;
    }
    if (!*slot || threshold < **slot)
      *slot = threshold;
  }
};

struct WaitcntScoreboard {
  CounterState vmem;
  CounterState lgkm;
  CounterState vscnt;
  DenseMap<Value, Ticket> valueTickets;

  bool merge(const WaitcntScoreboard &other) {
    bool changed = false;
    changed |= vmem.merge(other.vmem);
    changed |= lgkm.merge(other.lgkm);
    changed |= vscnt.merge(other.vscnt);
    for (auto [value, ticket] : other.valueTickets) {
      auto [it, inserted] = valueTickets.try_emplace(value, ticket);
      if (inserted) {
        changed = true;
        continue;
      }
      if (ticket.value < it->second.value) {
        it->second = ticket;
        changed = true;
      }
    }
    return changed;
  }
};

static bool isWaveMachineOp(Operation *op, StringRef name) {
  return op->getName().getStringRef() == ("wavemachine." + name).str();
}

static bool isWaveMachineOp(Operation *op) {
  return op->getName().getStringRef().starts_with("wavemachine.");
}

static bool isVALU(Operation *op) {
  StringRef name = op->getName().getStringRef();
  return name == "wavemachine.v_mbcnt_lo" ||
         name == "wavemachine.v_mov_b32_tuple" ||
         name == "wavemachine.v_add_u32" ||
         name == "wavemachine.v_and_b32" ||
         name == "wavemachine.v_or_b32" ||
         name == "wavemachine.v_xor_b32" ||
         name == "wavemachine.v_lshlrev_b32" ||
         name == "wavemachine.v_cmp_eq_u32" ||
         name == "wavemachine.v_cmp_ne_u32" ||
         name == "wavemachine.v_cmp_lt_u32" ||
         name == "wavemachine.v_cmp_le_u32" ||
         name == "wavemachine.v_cmp_gt_u32" ||
         name == "wavemachine.v_cmp_ge_u32" ||
         name == "wavemachine.v_readfirstlane_b32" ||
         name == "wavemachine.wmma_i32_16x16x16_iu8" ||
         name == "wavemachine.wmma_f32_16x16x16_f16";
}

static bool isSMEMLoad(Operation *op) {
  return isWaveMachineOp(op, "s_load_b32") || isWaveMachineOp(op, "s_load_b64");
}

static bool isVMEMLoad(Operation *op) {
  return isWaveMachineOp(op, "global_load_b32");
}

static bool isVMEMStore(Operation *op) {
  return isWaveMachineOp(op, "global_store_b32") ||
         isWaveMachineOp(op, "global_store_tuple_b32");
}

static bool isWaitcnt(Operation *op) {
  return isWaveMachineOp(op, "s_waitcnt") ||
         isWaveMachineOp(op, "s_waitcnt_vscnt");
}

static bool hasMemoryTicket(Operation *op) {
  return isSMEMLoad(op) || isVMEMLoad(op) || isVMEMStore(op);
}

static unsigned encodeWaitcnt(std::optional<unsigned> vmcnt,
                              std::optional<unsigned> lgkmcnt) {
  llvm::AMDGPU::IsaVersion gfx11{11, 0, 0};
  return llvm::AMDGPU::encodeWaitcnt(
      gfx11, vmcnt.value_or(~0u), /*expcnt=*/~0u, lgkmcnt.value_or(~0u));
}

static wavemachine::ImmType getImmType(MLIRContext *ctx) {
  return wavemachine::ImmType::get(ctx);
}

static Operation *createWMOp(OpBuilder &builder, Location loc, StringRef name,
                             ValueRange operands, TypeRange resultTypes,
                             ArrayRef<NamedAttribute> attrs = {}) {
  std::string opName = ("wavemachine." + name).str();
  OperationState state(loc, opName);
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

static std::optional<unsigned> getImmediate(Value value) {
  Operation *def = value.getDefiningOp();
  if (!def || !isWaveMachineOp(def, "imm"))
    return std::nullopt;
  return static_cast<unsigned>(
      def->getAttrOfType<IntegerAttr>("value").getInt());
}

static WaitRequirement
computeRequirement(Operation *op, const WaitcntScoreboard &scoreboard) {
  WaitRequirement requirement;
  for (Value operand : op->getOperands()) {
    auto it = scoreboard.valueTickets.find(operand);
    if (it == scoreboard.valueTickets.end())
      continue;
    const CounterState *counter = nullptr;
    switch (it->second.counter) {
    case CounterKind::Vmem:
      counter = &scoreboard.vmem;
      break;
    case CounterKind::Lgkm:
      counter = &scoreboard.lgkm;
      break;
    case CounterKind::Vscnt:
      counter = &scoreboard.vscnt;
      break;
    }
    if (auto wait = counter->computeWait(it->second.value))
      requirement.add(it->second.counter, *wait);
  }
  if (isWaveMachineOp(op, "s_endpgm")) {
    if (auto wait = scoreboard.vscnt.computeWait(scoreboard.vscnt.lastTicket))
      requirement.add(CounterKind::Vscnt, *wait);
  }
  return requirement;
}

static void observeRequirement(WaitcntScoreboard &scoreboard,
                               const WaitRequirement &requirement) {
  if (requirement.vmcnt)
    scoreboard.vmem.observeWait(*requirement.vmcnt);
  if (requirement.lgkmcnt)
    scoreboard.lgkm.observeWait(*requirement.lgkmcnt);
  if (requirement.vscnt)
    scoreboard.vscnt.observeWait(*requirement.vscnt);
}

static void observeExistingWait(Operation *op, WaitcntScoreboard &scoreboard) {
  if (isWaveMachineOp(op, "s_waitcnt")) {
    auto imm = getImmediate(op->getOperand(0));
    if (!imm)
      return;
    llvm::AMDGPU::IsaVersion gfx11{11, 0, 0};
    unsigned vm = 0;
    unsigned exp = 0;
    unsigned lg = 0;
    llvm::AMDGPU::decodeWaitcnt(gfx11, *imm, vm, exp, lg);
    scoreboard.vmem.observeWait(vm);
    scoreboard.lgkm.observeWait(lg);
    return;
  }
  if (isWaveMachineOp(op, "s_waitcnt_vscnt")) {
    auto imm = getImmediate(op->getOperand(0));
    if (imm)
      scoreboard.vscnt.observeWait(*imm);
  }
}

static void propagateTicket(WaitcntScoreboard &scoreboard, Value src,
                            Value dst) {
  if (!src || !dst)
    return;
  auto it = scoreboard.valueTickets.find(src);
  if (it != scoreboard.valueTickets.end())
    scoreboard.valueTickets[dst] = it->second;
}

static void assignOperationTickets(func::FuncOp func,
                                   DenseMap<Operation *, Ticket> &tickets) {
  int64_t vmem = -1;
  int64_t lgkm = -1;
  int64_t vscnt = -1;
  func.walk([&](Operation *op) {
    if (!hasMemoryTicket(op))
      return;
    if (isSMEMLoad(op)) {
      tickets[op] = Ticket{CounterKind::Lgkm, ++lgkm};
      return;
    }
    if (isVMEMLoad(op)) {
      tickets[op] = Ticket{CounterKind::Vmem, ++vmem};
      return;
    }
    if (isVMEMStore(op))
      tickets[op] = Ticket{CounterKind::Vscnt, ++vscnt};
  });
}

static void emitWaits(OpBuilder &builder, Location loc,
                      const WaitRequirement &requirement) {
  if (requirement.vmcnt || requirement.lgkmcnt) {
    unsigned encoded = encodeWaitcnt(requirement.vmcnt, requirement.lgkmcnt);
    createInstrNoResult(builder, loc, "s_waitcnt",
                        createImm(builder, loc, encoded));
  }
  if (requirement.vscnt) {
    createInstrNoResult(builder, loc, "s_waitcnt_vscnt",
                        createImm(builder, loc, *requirement.vscnt));
  }
}

static void propagateBranchOperands(Operation *terminator, Block *successor,
                                    WaitcntScoreboard &scoreboard);

static LogicalResult validateWaveMachineOp(Operation *op) {
  if (!isWaveMachineOp(op))
    return success();
  if (auto func = op->getParentOfType<func::FuncOp>();
      func && func->hasAttr("wave.kernel") && isWaveMachineOp(op, "arg"))
    return op->emitError("wavemachine-insert-ticket-waits expects "
                         "ABI-lowered kernel arguments");
  if (isSMEMLoad(op) && !op->getAttrOfType<StringAttr>("base"))
    return op->emitError("wavemachine-insert-ticket-waits expects scalar "
                         "memory loads to carry a base register attribute");
  return success();
}

static void observeTicket(Operation *op,
                          const DenseMap<Operation *, Ticket> &operationTickets,
                          WaitcntScoreboard &scoreboard) {
  auto it = operationTickets.find(op);
  if (it == operationTickets.end())
    return;
  const Ticket &ticket = it->second;
  switch (ticket.counter) {
  case CounterKind::Vmem:
    scoreboard.vmem.observeIssue(ticket.value);
    break;
  case CounterKind::Lgkm:
    scoreboard.lgkm.observeIssue(ticket.value);
    break;
  case CounterKind::Vscnt:
    scoreboard.vscnt.observeIssue(ticket.value);
    break;
  }
  for (Value result : op->getResults())
    scoreboard.valueTickets[result] = ticket;
}

static LogicalResult
transferBlock(Block &block, const WaitcntScoreboard &input,
              const DenseMap<Operation *, Ticket> &operationTickets,
              WaitcntScoreboard &output) {
  output = input;
  for (Operation &op : block) {
    if (failed(validateWaveMachineOp(&op)))
      return failure();
    if (!isWaveMachineOp(&op))
      continue;
    if (isWaitcnt(&op)) {
      observeExistingWait(&op, output);
      continue;
    }
    WaitRequirement requirement = computeRequirement(&op, output);
    observeRequirement(output, requirement);
    observeTicket(&op, operationTickets, output);
  }
  return success();
}

static LogicalResult
processFunctionCFG(func::FuncOp func,
                   const DenseMap<Operation *, Ticket> &operationTickets) {
  DenseMap<Block *, WaitcntScoreboard> blockInputs;
  DenseMap<Block *, WaitcntScoreboard> blockOutputs;
  SmallVector<Block *> blocks;
  for (Block &block : func.getBody())
    blocks.push_back(&block);

  bool changed = true;
  for (unsigned iteration = 0; changed && iteration != 128; ++iteration) {
    changed = false;
    for (Block *block : blocks) {
      WaitcntScoreboard output;
      if (failed(transferBlock(*block, blockInputs[block], operationTickets,
                               output)))
        return failure();
      if (blockOutputs[block].merge(output))
        changed = true;

      Operation *terminator = block->getTerminator();
      for (Block *successor : terminator->getSuccessors()) {
        WaitcntScoreboard successorInput = output;
        propagateBranchOperands(terminator, successor, successorInput);
        if (blockInputs[successor].merge(successorInput))
          changed = true;
      }
    }
  }
  if (changed)
    return func.emitError("wavemachine-insert-ticket-waits failed to converge");

  OpBuilder builder(func.getContext());
  for (Block *block : blocks) {
    WaitcntScoreboard scoreboard = blockInputs[block];
    for (Operation &op : llvm::make_early_inc_range(*block)) {
      if (!isWaveMachineOp(&op))
        continue;
      if (isWaitcnt(&op)) {
        observeExistingWait(&op, scoreboard);
        continue;
      }
      WaitRequirement requirement = computeRequirement(&op, scoreboard);
      if (requirement.hasWait()) {
        builder.setInsertionPoint(&op);
        emitWaits(builder, op.getLoc(), requirement);
        if (requirement.lgkmcnt && isVALU(&op))
          createInstrNoResult(builder, op.getLoc(), "s_delay_alu",
                              createImm(builder, op.getLoc(), 1));
        observeRequirement(scoreboard, requirement);
      }
      observeTicket(&op, operationTickets, scoreboard);
    }
  }
  return success();
}

static void propagateBranchOperands(Operation *terminator, Block *successor,
                                    WaitcntScoreboard &scoreboard) {
  bool mappedSuccessorOperands = false;
  if (auto branch = dyn_cast<BranchOpInterface>(terminator)) {
    for (auto [index, target] : llvm::enumerate(branch->getSuccessors())) {
      if (target != successor)
        continue;
      SuccessorOperands operands = branch.getSuccessorOperands(index);
      for (auto [argIndex, arg] : llvm::enumerate(successor->getArguments())) {
        if (argIndex >= operands.size())
          break;
        propagateTicket(scoreboard, operands[argIndex], arg);
      }
      mappedSuccessorOperands = true;
    }
  }
  if (!mappedSuccessorOperands && terminator->getNumSuccessors() == 1 &&
      terminator->getSuccessor(0) == successor &&
      terminator->getNumOperands() >= successor->getNumArguments()) {
    for (auto [argIndex, arg] : llvm::enumerate(successor->getArguments()))
      propagateTicket(scoreboard, terminator->getOperand(argIndex), arg);
  }
}

struct WaveMachineTicketWaitsPass
    : public wave::impl::WaveMachineTicketWaitsBase<WaveMachineTicketWaitsPass> {
  void runOnOperation() override {
    ModuleOp module = getOperation();
    for (func::FuncOp func : module.getOps<func::FuncOp>()) {
      DenseMap<Operation *, Ticket> operationTickets;
      assignOperationTickets(func, operationTickets);
      if (failed(processFunctionCFG(func, operationTickets)))
        return signalPassFailure();
    }
  }
};

} // namespace

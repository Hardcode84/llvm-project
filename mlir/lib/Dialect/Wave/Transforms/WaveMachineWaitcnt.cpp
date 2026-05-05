//===- WaveMachineWaitcnt.cpp - WaveMachine waitcnt insertion ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Wave/Transforms/Passes.h"

#include "Utils/AMDGPUBaseInfo.h"
#include "mlir/Analysis/DataFlow/DenseAnalysis.h"
#include "mlir/Analysis/DataFlow/Utils.h"
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
using namespace mlir::dataflow;
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
    if (lastTicket < 0)
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

class WaitcntState : public AbstractDenseLattice {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(WaitcntState)

  using AbstractDenseLattice::AbstractDenseLattice;

  ChangeResult join(const AbstractDenseLattice &rhs) override {
    const auto &other = static_cast<const WaitcntState &>(rhs);
    WaitcntScoreboard merged = scoreboard;
    if (!merged.merge(other.scoreboard))
      return ChangeResult::NoChange;
    scoreboard = std::move(merged);
    return ChangeResult::Change;
  }

  ChangeResult joinScoreboard(const WaitcntScoreboard &rhs) {
    WaitcntScoreboard merged = scoreboard;
    if (!merged.merge(rhs))
      return ChangeResult::NoChange;
    scoreboard = std::move(merged);
    return ChangeResult::Change;
  }

  void print(raw_ostream &os) const override {
    os << "vmem=" << scoreboard.vmem.lastTicket << " lgkm="
       << scoreboard.lgkm.lastTicket << " vscnt="
       << scoreboard.vscnt.lastTicket << " values="
       << scoreboard.valueTickets.size();
  }

  ChangeResult reset() {
    if (scoreboard.vmem.lastTicket == -1 &&
        scoreboard.lgkm.lastTicket == -1 &&
        scoreboard.vscnt.lastTicket == -1 &&
        scoreboard.valueTickets.empty())
      return ChangeResult::NoChange;
    scoreboard = WaitcntScoreboard();
    return ChangeResult::Change;
  }

  WaitcntScoreboard &mutate() { return scoreboard; }
  const WaitcntScoreboard &get() const { return scoreboard; }

private:
  WaitcntScoreboard scoreboard;
};

static bool isWaveMachineOp(Operation *op, StringRef name) {
  return op->getName().getStringRef() == ("wavemachine." + name).str();
}

static bool isWaveMachineOp(Operation *op) {
  return op->getName().getStringRef().starts_with("wavemachine.");
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

static unsigned counterIndex(CounterKind counter) {
  switch (counter) {
  case CounterKind::Vmem:
    return 0;
  case CounterKind::Lgkm:
    return 1;
  case CounterKind::Vscnt:
    return 2;
  }
  llvm_unreachable("unknown counter");
}

static void propagateTicket(WaitcntScoreboard &scoreboard, Value src, Value dst,
                            ArrayRef<int64_t> ticketShift = {}) {
  if (!src || !dst)
    return;
  auto it = scoreboard.valueTickets.find(src);
  if (it == scoreboard.valueTickets.end())
    return;
  Ticket ticket = it->second;
  if (!ticketShift.empty()) {
    ticket.value -= ticketShift[counterIndex(ticket.counter)];
    ticket.value = std::max<int64_t>(ticket.value, -64);
  }
  scoreboard.valueTickets[dst] = ticket;
}

static void propagateTickets(WaitcntScoreboard &scoreboard, ValueRange sources,
                             ValueRange destinations,
                             ArrayRef<int64_t> ticketShift = {}) {
  for (auto [src, dst] : llvm::zip_equal(sources, destinations))
    propagateTicket(scoreboard, src, dst, ticketShift);
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
                                    WaitcntScoreboard &scoreboard,
                                    ArrayRef<int64_t> ticketShift = {});

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

static void collectBlocks(Region &region, SmallVectorImpl<Block *> &blocks) {
  for (Block &block : region) {
    blocks.push_back(&block);
    for (Operation &op : block)
      for (Region &nested : op.getRegions())
        collectBlocks(nested, blocks);
  }
}

static void countIssuesInBlock(Block *block, int64_t (&counts)[3]) {
  for (Operation &op : *block) {
    if (isVMEMLoad(&op))
      ++counts[counterIndex(CounterKind::Vmem)];
    if (isSMEMLoad(&op))
      ++counts[counterIndex(CounterKind::Lgkm)];
    if (isVMEMStore(&op))
      ++counts[counterIndex(CounterKind::Vscnt)];
  }
}

static bool isBackedge(Block *source, Block *dest,
                       const DenseMap<Block *, unsigned> &blockOrder) {
  auto sourceIt = blockOrder.find(source);
  auto destIt = blockOrder.find(dest);
  if (sourceIt == blockOrder.end() || destIt == blockOrder.end())
    return false;
  return destIt->second <= sourceIt->second;
}

static void computeTicketShift(Block *source, Block *dest,
                               const DenseMap<Block *, unsigned> &blockOrder,
                               int64_t (&shift)[3]) {
  shift[0] = shift[1] = shift[2] = 0;
  if (!isBackedge(source, dest, blockOrder))
    return;
  countIssuesInBlock(dest, shift);
}

class WaitcntAnalysis : public DenseForwardDataFlowAnalysis<WaitcntState> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(WaitcntAnalysis)

  WaitcntAnalysis(DataFlowSolver &solver,
                  const DenseMap<Operation *, Ticket> &operationTickets,
                  const DenseMap<Block *, unsigned> &blockOrder)
      : DenseForwardDataFlowAnalysis(solver),
        operationTickets(operationTickets), blockOrder(blockOrder) {}

  LogicalResult initialize(Operation *top) override {
    auto markOperation = [&](Operation *op) {
      for (Region &region : op->getRegions()) {
        for (Block &block : region) {
          auto *blockLive =
              getOrCreate<Executable>(getProgramPointBefore(&block));
          propagateIfChanged(blockLive, blockLive->setToLive());
          Operation *terminator = block.getTerminator();
          if (!terminator)
            continue;
          for (Block *successor : terminator->getSuccessors()) {
            auto *edgeLive = getOrCreate<Executable>(
                getLatticeAnchor<CFGEdge>(&block, successor));
            propagateIfChanged(edgeLive, edgeLive->setToLive());
          }
        }
      }
    };
    markOperation(top);
    top->walk(markOperation);
    return DenseForwardDataFlowAnalysis<WaitcntState>::initialize(top);
  }

  void setToEntryState(WaitcntState *lattice) override {
    propagateIfChanged(lattice, lattice->reset());
  }

  LogicalResult visitOperation(Operation *op, const WaitcntState &before,
                               WaitcntState *after) override {
    if (failed(validateWaveMachineOp(op)))
      return failure();

    WaitcntState next = before;
    WaitcntScoreboard &scoreboard = next.mutate();
    if (isWaitcnt(op)) {
      observeExistingWait(op, scoreboard);
    } else {
      observeTicket(op, operationTickets, scoreboard);
    }

    propagateIfChanged(after, after->join(next));
    markCFGSuccessorsLive(op, next.get());
    return success();
  }

  void visitBlockTransfer(Block *block, ProgramPoint *point, Block *predecessor,
                          const WaitcntState &before,
                          WaitcntState *after) override {
    WaitcntState next = before;
    int64_t shift[3];
    computeTicketShift(predecessor, block, blockOrder, shift);
    propagateBranchOperands(predecessor->getTerminator(), block, next.mutate(),
                            shift);
    propagateIfChanged(after, after->join(next));
  }

  void visitRegionBranchControlFlowTransfer(
      RegionBranchOpInterface branch, std::optional<unsigned> regionFrom,
      std::optional<unsigned> regionTo, const WaitcntState &before,
      WaitcntState *after) override {
    WaitcntState next = before;
    WaitcntScoreboard &scoreboard = next.mutate();
    RegionSuccessor successor =
        regionTo ? RegionSuccessor(&branch->getRegion(*regionTo))
                 : RegionSuccessor::parent();

    SmallVector<Value> sources;
    Block *sourceBlock = nullptr;
    if (regionFrom) {
      Operation *terminator =
          branch->getRegion(*regionFrom).front().getTerminator();
      sourceBlock = terminator->getBlock();
      if (auto regionTerm =
              dyn_cast<RegionBranchTerminatorOpInterface>(terminator))
        llvm::append_range(sources, regionTerm.getSuccessorOperands(successor));
    } else {
      llvm::append_range(sources, branch.getEntrySuccessorOperands(successor));
    }

    int64_t shift[3] = {0, 0, 0};
    if (regionFrom && regionTo && sourceBlock)
      computeTicketShift(sourceBlock, &branch->getRegion(*regionTo).front(),
                         blockOrder, shift);

    propagateTickets(scoreboard, sources, branch.getSuccessorInputs(successor),
                     shift);

    propagateIfChanged(after, after->join(next));
  }

private:
  void markCFGSuccessorsLive(Operation *op,
                             const WaitcntScoreboard &scoreboard) {
    if (op->getNumSuccessors() == 0)
      return;
    Block *source = op->getBlock();
    if (!source)
      return;
    for (Block *successor : op->getSuccessors()) {
      WaitcntScoreboard successorState = scoreboard;
      int64_t shift[3];
      computeTicketShift(source, successor, blockOrder, shift);
      propagateBranchOperands(op, successor, successorState, shift);
      auto *blockState =
          getLattice(getProgramPointBefore(successor));
      propagateIfChanged(blockState, blockState->joinScoreboard(successorState));
      auto *blockLive = getOrCreate<Executable>(getProgramPointBefore(successor));
      propagateIfChanged(blockLive, blockLive->setToLive());
      auto *edgeLive = getOrCreate<Executable>(
          getLatticeAnchor<CFGEdge>(source, successor));
      propagateIfChanged(edgeLive, edgeLive->setToLive());
    }
  }

  const DenseMap<Operation *, Ticket> &operationTickets;
  const DenseMap<Block *, unsigned> &blockOrder;
};

static void propagateBranchOperands(Operation *terminator, Block *successor,
                                    WaitcntScoreboard &scoreboard,
                                    ArrayRef<int64_t> ticketShift) {
  bool mappedSuccessorOperands = false;
  if (auto branch = dyn_cast<BranchOpInterface>(terminator)) {
    for (auto [index, target] : llvm::enumerate(branch->getSuccessors())) {
      if (target != successor)
        continue;
      SuccessorOperands operands = branch.getSuccessorOperands(index);
      for (auto [argIndex, arg] : llvm::enumerate(successor->getArguments())) {
        if (argIndex >= operands.size())
          break;
        propagateTicket(scoreboard, operands[argIndex], arg, ticketShift);
      }
      mappedSuccessorOperands = true;
    }
  }
  if (!mappedSuccessorOperands && terminator->getNumSuccessors() == 1 &&
      terminator->getSuccessor(0) == successor &&
      terminator->getNumOperands() >= successor->getNumArguments()) {
    for (auto [argIndex, arg] : llvm::enumerate(successor->getArguments()))
      propagateTicket(scoreboard, terminator->getOperand(argIndex), arg,
                      ticketShift);
  }
}

static WaitcntScoreboard
getEffectiveStateBefore(Operation *op, DataFlowSolver &solver,
                        const DenseMap<Block *, unsigned> &blockOrder) {
  WaitcntScoreboard effective;
  if (auto *state = solver.lookupState<WaitcntState>(
          solver.getProgramPointBefore(op)))
    effective.merge(state->get());

  Block *block = op->getBlock();
  if (!block)
    return effective;
  if (auto *blockState = solver.lookupState<WaitcntState>(
          solver.getProgramPointBefore(block)))
    effective.merge(blockState->get());
  if (block->isEntryBlock())
    return effective;

  for (Block *predecessor : block->getPredecessors()) {
    Operation *terminator = predecessor->getTerminator();
    auto *predState = solver.lookupState<WaitcntState>(
        solver.getProgramPointAfter(terminator));
    if (!predState)
      continue;
    WaitcntScoreboard predEffective = predState->get();
    int64_t shift[3];
    computeTicketShift(predecessor, block, blockOrder, shift);
    propagateBranchOperands(terminator, block, predEffective, shift);
    effective.merge(predEffective);
  }
  return effective;
}

struct WaveMachineTicketWaitsPass
    : public wave::impl::WaveMachineTicketWaitsBase<WaveMachineTicketWaitsPass> {
  void runOnOperation() override {
    ModuleOp module = getOperation();
    for (func::FuncOp func : module.getOps<func::FuncOp>()) {
      DenseMap<Operation *, Ticket> operationTickets;
      DenseMap<Block *, unsigned> blockOrder;
      assignOperationTickets(func, operationTickets);
      SmallVector<Block *> blocks;
      collectBlocks(func.getBody(), blocks);
      for (auto [index, block] : llvm::enumerate(blocks))
        blockOrder[block] = index;

      DataFlowSolver solver;
      loadBaselineAnalyses(solver);
      solver.load<WaitcntAnalysis>(operationTickets, blockOrder);
      if (failed(solver.initializeAndRun(func)))
        return signalPassFailure();

      OpBuilder builder(func.getContext());
      SmallVector<Operation *> ops;
      func.walk([&](Operation *op) {
        if (isWaveMachineOp(op) && !isWaitcnt(op))
          ops.push_back(op);
      });

      for (Operation *op : ops) {
        WaitcntScoreboard effective =
            getEffectiveStateBefore(op, solver, blockOrder);
        WaitRequirement requirement = computeRequirement(op, effective);
        if (!requirement.hasWait())
          continue;
        builder.setInsertionPoint(op);
        emitWaits(builder, op->getLoc(), requirement);
      }
    }
  }
};

} // namespace

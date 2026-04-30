//===- AMDGPU.cpp - Wave to AMDGPU backend --------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Target/Wave/AMDGPU.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Wave/IR/Wave.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringSwitch.h"
#include <algorithm>
#include <optional>

using namespace mlir;
using namespace mlir::wave;

namespace {

enum class RegClass { SGPR, VGPR };

struct VirtualReg {
  unsigned id = 0;
  RegClass regClass = RegClass::VGPR;
};

struct Operand {
  enum class Kind { Reg, Imm, Label } kind = Kind::Imm;
  unsigned regId = 0;
  int64_t immValue = 0;
  std::string labelValue;

  static Operand makeReg(unsigned id) {
    Operand operand;
    operand.kind = Kind::Reg;
    operand.regId = id;
    return operand;
  }

  static Operand makeImm(int64_t value) {
    Operand operand;
    operand.kind = Kind::Imm;
    operand.immValue = value;
    return operand;
  }

  static Operand makeLabel(StringRef value) {
    Operand operand;
    operand.kind = Kind::Label;
    operand.labelValue = value.str();
    return operand;
  }
};

enum class MachineOpcode {
  Label,
  Comment,
  VMbcntLo,
  VAddU32,
  VAndB32,
  VOrB32,
  VXorB32,
  VLshlRevB32,
  VCmpEqU32,
  VCmpNeU32,
  VCmpLtU32,
  VCmpLeU32,
  VCmpGtU32,
  VCmpGeU32,
  SMovB32,
  SAndSaveExecB32,
  SCBranchExecZ,
  SMovExecLo,
  VReadFirstLaneB32,
  SSetPcB64
};

struct MachineInstr {
  MachineOpcode opcode = MachineOpcode::Comment;
  SmallVector<unsigned, 1> defs;
  SmallVector<Operand, 3> operands;
  std::string text;
};

struct LiveInterval {
  unsigned reg = 0;
  RegClass regClass = RegClass::VGPR;
  unsigned start = std::numeric_limits<unsigned>::max();
  unsigned end = 0;
};

class WaveAMDGPUEmitter {
public:
  explicit WaveAMDGPUEmitter(raw_ostream &os) : os(os) {}

  LogicalResult emit(Operation *op) {
    auto module = dyn_cast<ModuleOp>(op);
    if (!module)
      return op->emitError("wave AMDGPU backend expects a module operation");

    os << "\t.text\n";
    os << "\t.amdgcn_target \"amdgcn-amd-amdhsa--gfx1100\"\n";
    for (func::FuncOp func : module.getOps<func::FuncOp>()) {
      if (failed(emitFunction(func)))
        return failure();
    }
    return success();
  }

private:
  raw_ostream &os;
  DenseMap<Value, Operand> values;
  SmallVector<VirtualReg> virtualRegs;
  SmallVector<MachineInstr> instructions;
  DenseMap<unsigned, unsigned> allocation;
  unsigned nextLabel = 0;
  unsigned indent = 1;

  unsigned createVirtualReg(RegClass regClass) {
    unsigned id = virtualRegs.size();
    virtualRegs.push_back(VirtualReg{id, regClass});
    return id;
  }

  std::string makeLabel(StringRef stem) {
    return (Twine(".Lwave_") + stem + "_" + Twine(nextLabel++)).str();
  }

  void emitLine(StringRef line) {
    for (unsigned i = 0; i < indent; ++i)
      os << '\t';
    os << line << '\n';
  }

  void emitLine(const Twine &line) {
    SmallString<128> storage;
    emitLine(line.toStringRef(storage));
  }

  LogicalResult emitFunction(func::FuncOp func) {
    values.clear();
    virtualRegs.clear();
    instructions.clear();
    allocation.clear();

    // Ordinary function arguments are scalar-uniform in the Wave source model.
    // A future ABI layer will assign the exact physical user SGPRs; for now we
    // model that contract directly in the machine IR.
    for (BlockArgument arg : func.getArguments()) {
      RegClass regClass = isa<SimdType>(arg.getType()) ? RegClass::VGPR
                                                       : RegClass::SGPR;
      values[arg] = Operand::makeReg(createVirtualReg(regClass));
    }

    if (!func.getBody().hasOneBlock())
      return func.emitError("wave AMDGPU backend only supports one-block funcs");
    for (Operation &op : func.getBody().front()) {
      if (failed(selectOperation(&op)))
        return failure();
    }

    if (failed(allocateRegisters(func)))
      return failure();

    os << "\n\t.globl\t" << func.getSymName() << "\n";
    os << "\t.p2align\t8\n";
    os << "\t.type\t" << func.getSymName() << ",@function\n";
    os << func.getSymName() << ":\n";

    emitLine(StringRef("; wave backend: virtual registers allocated by linear scan"));
    for (const MachineInstr &mi : instructions) {
      if (failed(emitMachineInstr(mi)))
        return failure();
    }

    os << "\t.size\t" << func.getSymName() << ", .-" << func.getSymName()
       << "\n";
    return success();
  }

  FailureOr<Operand> lookup(Value value) {
    auto it = values.find(value);
    if (it == values.end())
      return failure();
    return it->second;
  }

  Operand expect(Value value, Operation *user) {
    FailureOr<Operand> result = lookup(value);
    if (failed(result)) {
      user->emitError("value has no backend location");
      return Operand::makeImm(0);
    }
    return *result;
  }

  void addInstr(MachineOpcode opcode, ArrayRef<unsigned> defs,
                ArrayRef<Operand> operands = {}) {
    MachineInstr mi;
    mi.opcode = opcode;
    mi.defs.append(defs.begin(), defs.end());
    mi.operands.append(operands.begin(), operands.end());
    instructions.push_back(std::move(mi));
  }

  void addLabel(StringRef label) {
    MachineInstr mi;
    mi.opcode = MachineOpcode::Label;
    mi.text = label.str();
    instructions.push_back(std::move(mi));
  }

  void addComment(const Twine &text) {
    MachineInstr mi;
    mi.opcode = MachineOpcode::Comment;
    mi.text = text.str();
    instructions.push_back(std::move(mi));
  }

  LogicalResult selectOperation(Operation *op) {
    if (auto constant = dyn_cast<arith::ConstantIntOp>(op))
      return selectConstant(constant);
    if (auto laneId = dyn_cast<LaneIdOp>(op))
      return selectLaneId(laneId);
    if (auto splat = dyn_cast<SplatOp>(op))
      return selectSplat(splat);
    if (auto binary = dyn_cast<BinaryOp>(op))
      return selectBinary(binary);
    if (auto cmp = dyn_cast<CmpIOp>(op))
      return selectCmp(cmp);
    if (auto ballot = dyn_cast<BallotOp>(op))
      return selectBallot(ballot);
    if (auto readFirst = dyn_cast<ReadFirstOp>(op))
      return selectReadFirst(readFirst);
    if (auto where = dyn_cast<WhereOp>(op))
      return selectWhere(where);
    if (auto store = dyn_cast<StoreOp>(op))
      return selectStore(store);
    if (auto ret = dyn_cast<func::ReturnOp>(op))
      return selectReturn(ret);
    if (isa<YieldOp>(op))
      return success();

    return op->emitError("unsupported operation in Wave AMDGPU backend");
  }

  LogicalResult selectConstant(arith::ConstantIntOp op) {
    values[op.getResult()] = Operand::makeImm(op.value());
    return success();
  }

  LogicalResult selectLaneId(LaneIdOp op) {
    auto simdType = cast<SimdType>(op.getType());
    if (!simdType.getElementType().isInteger(32) || simdType.getWidth() != 32)
      return op.emitError("backend supports only !wave.simd<i32, 32> lane_id");

    unsigned dst = createVirtualReg(RegClass::VGPR);
    addInstr(MachineOpcode::VMbcntLo, dst);
    values[op.getResult()] = Operand::makeReg(dst);
    return success();
  }

  LogicalResult selectSplat(SplatOp op) {
    values[op.getResult()] = expect(op.getSource(), op);
    return success();
  }

  LogicalResult selectBinary(BinaryOp op) {
    unsigned dst = createVirtualReg(RegClass::VGPR);
    MachineOpcode opcode =
        llvm::StringSwitch<MachineOpcode>(op.getKind())
            .Case("addi", MachineOpcode::VAddU32)
            .Case("andi", MachineOpcode::VAndB32)
            .Case("ori", MachineOpcode::VOrB32)
            .Case("xori", MachineOpcode::VXorB32)
            .Case("shli", MachineOpcode::VLshlRevB32)
            .Default(MachineOpcode::Comment);
    if (opcode == MachineOpcode::Comment)
      return op.emitError("unsupported wave.binary kind");
    addInstr(opcode, dst, {expect(op.getLhs(), op), expect(op.getRhs(), op)});
    values[op.getResult()] = Operand::makeReg(dst);
    return success();
  }

  LogicalResult selectCmp(CmpIOp op) {
    auto maskType = cast<MaskType>(op.getType());
    if (maskType.getWidth() != 32)
      return op.emitError("backend supports only !wave.mask<32>");

    unsigned dst = createVirtualReg(RegClass::SGPR);
    MachineOpcode cmp =
        llvm::StringSwitch<MachineOpcode>(
            stringifyCmpIPredicate(op.getPredicate()))
            .Case("eq", MachineOpcode::VCmpEqU32)
            .Case("ne", MachineOpcode::VCmpNeU32)
            .Case("ult", MachineOpcode::VCmpLtU32)
            .Case("ule", MachineOpcode::VCmpLeU32)
            .Case("ugt", MachineOpcode::VCmpGtU32)
            .Case("uge", MachineOpcode::VCmpGeU32)
            .Default(MachineOpcode::Comment);
    if (cmp == MachineOpcode::Comment)
      return op.emitError("unsupported wave.cmpi predicate");
    addInstr(cmp, dst, {expect(op.getLhs(), op), expect(op.getRhs(), op)});
    values[op.getResult()] = Operand::makeReg(dst);
    return success();
  }

  LogicalResult selectBallot(BallotOp op) {
    // In the MVP backend a mask is already represented as an EXEC-width SGPR
    // bit mask, so materializing it as an integer is just an alias.
    values[op.getResult()] = expect(op.getMask(), op);
    return success();
  }

  LogicalResult selectReadFirst(ReadFirstOp op) {
    Operand src = expect(op.getSource(), op);
    if (src.kind == Operand::Kind::Reg &&
        virtualRegs[src.regId].regClass == RegClass::SGPR) {
      values[op.getResult()] = src;
      return success();
    }
    unsigned dst = createVirtualReg(RegClass::SGPR);
    addInstr(MachineOpcode::VReadFirstLaneB32, dst, src);
    values[op.getResult()] = Operand::makeReg(dst);
    return success();
  }

  LogicalResult selectStore(StoreOp op) {
    // This backend is intentionally text-only for now. Record where a real
    // memory backend would select a flat/global store.
    MachineInstr mi;
    mi.opcode = MachineOpcode::Comment;
    mi.operands.push_back(expect(op.getValue(), op));
    mi.text = "wave.store";
    instructions.push_back(std::move(mi));
    return success();
  }

  LogicalResult selectWhere(WhereOp op) {
    unsigned savedExec = createVirtualReg(RegClass::SGPR);
    std::string endLabel = makeLabel("endif");
    addInstr(MachineOpcode::SAndSaveExecB32, savedExec,
             expect(op.getCondition(), op));
    addInstr(MachineOpcode::SCBranchExecZ, {}, Operand::makeLabel(endLabel));
    if (failed(selectRegion(op.getThenRegion())))
      return failure();
    addLabel(endLabel);
    addInstr(MachineOpcode::SMovExecLo, {}, Operand::makeReg(savedExec));
    if (!op.getElseRegion().empty()) {
      addComment("; otherwise region omitted in MVP backend");
    }
    return success();
  }

  LogicalResult selectRegion(Region &region) {
    if (!region.hasOneBlock())
      return failure();
    for (Operation &op : region.front()) {
      if (failed(selectOperation(&op)))
        return failure();
    }
    return success();
  }

  LogicalResult selectReturn(func::ReturnOp op) {
    if (op.getNumOperands() > 1)
      return op.emitError("backend supports at most one return value");
    if (op.getNumOperands() == 1) {
      Operand ret = expect(op.getOperand(0), op);
      if (ret.kind == Operand::Kind::Reg &&
          virtualRegs[ret.regId].regClass == RegClass::VGPR) {
        addInstr(MachineOpcode::VReadFirstLaneB32, createVirtualReg(RegClass::SGPR),
                 ret);
        unsigned returnReg = instructions.back().defs.front();
        addInstr(MachineOpcode::SMovB32, {}, {Operand::makeLabel("s0"),
                                              Operand::makeReg(returnReg)});
      } else {
        addInstr(MachineOpcode::SMovB32, {}, {Operand::makeLabel("s0"), ret});
      }
    }
    addInstr(MachineOpcode::SSetPcB64, {});
    return success();
  }

  static bool isRegOperand(const Operand &operand) {
    return operand.kind == Operand::Kind::Reg;
  }

  SmallVector<LiveInterval> computeLiveIntervals() {
    SmallVector<LiveInterval> intervals;
    intervals.reserve(virtualRegs.size());
    for (VirtualReg reg : virtualRegs)
      intervals.push_back(LiveInterval{reg.id, reg.regClass});

    auto touch = [&](unsigned reg, unsigned pos) {
      LiveInterval &interval = intervals[reg];
      interval.start = std::min(interval.start, pos);
      interval.end = std::max(interval.end, pos);
    };

    for (unsigned pos = 0, e = instructions.size(); pos != e; ++pos) {
      const MachineInstr &mi = instructions[pos];
      for (unsigned def : mi.defs)
        touch(def, pos);
      for (const Operand &operand : mi.operands) {
        if (isRegOperand(operand))
          touch(operand.regId, pos);
      }
    }

    for (LiveInterval &interval : intervals) {
      if (interval.start == std::numeric_limits<unsigned>::max())
        interval.start = interval.end = 0;
    }
    return intervals;
  }

  LogicalResult allocateRegisters(func::FuncOp func) {
    SmallVector<LiveInterval> intervals = computeLiveIntervals();
    llvm::stable_sort(intervals, [](const LiveInterval &lhs,
                                   const LiveInterval &rhs) {
      if (lhs.start != rhs.start)
        return lhs.start < rhs.start;
      return lhs.reg < rhs.reg;
    });

    if (failed(allocateClass(func, intervals, RegClass::VGPR, /*numPhys=*/32)))
      return failure();
    if (failed(allocateClass(func, intervals, RegClass::SGPR, /*numPhys=*/32)))
      return failure();
    return success();
  }

  LogicalResult allocateClass(func::FuncOp func, ArrayRef<LiveInterval> intervals,
                              RegClass regClass, unsigned numPhys) {
    SmallVector<LiveInterval> active;
    SmallVector<unsigned> freeRegs;
    for (unsigned i = 0; i != numPhys; ++i)
      freeRegs.push_back(numPhys - 1 - i);

    auto expireOld = [&](unsigned pos) {
      SmallVector<LiveInterval> stillActive;
      for (LiveInterval interval : active) {
        if (interval.end < pos) {
          freeRegs.push_back(allocation[interval.reg]);
        } else {
          stillActive.push_back(interval);
        }
      }
      active = std::move(stillActive);
    };

    for (LiveInterval interval : intervals) {
      if (interval.regClass != regClass)
        continue;
      expireOld(interval.start);
      if (freeRegs.empty())
        return func.emitError("wave backend ran out of physical registers");
      unsigned phys = freeRegs.pop_back_val();
      allocation[interval.reg] = phys;
      active.push_back(interval);
      llvm::sort(active, [](const LiveInterval &lhs, const LiveInterval &rhs) {
        return lhs.end < rhs.end;
      });
    }
    return success();
  }

  std::string physReg(unsigned reg) const {
    const VirtualReg &vreg = virtualRegs[reg];
    auto it = allocation.find(reg);
    assert(it != allocation.end() && "unallocated virtual register");
    return (vreg.regClass == RegClass::VGPR ? "v" : "s") + Twine(it->second).str();
  }

  std::string operandToString(const Operand &operand) const {
    switch (operand.kind) {
    case Operand::Kind::Reg:
      return physReg(operand.regId);
    case Operand::Kind::Imm:
      return Twine(operand.immValue).str();
    case Operand::Kind::Label:
      return operand.labelValue;
    }
    llvm_unreachable("unknown operand kind");
  }

  LogicalResult emitMachineInstr(const MachineInstr &mi) {
    auto def = [&]() { return physReg(mi.defs.front()); };
    auto op = [&](unsigned i) { return operandToString(mi.operands[i]); };

    switch (mi.opcode) {
    case MachineOpcode::Label:
      os << mi.text << ":\n";
      return success();
    case MachineOpcode::Comment:
      if (!mi.operands.empty())
        emitLine(Twine("; ") + mi.text + " " + op(0));
      else
        emitLine(Twine("; ") + mi.text);
      return success();
    case MachineOpcode::VMbcntLo:
      emitLine(Twine("v_mbcnt_lo_u32_b32 ") + def() + ", -1, 0");
      return success();
    case MachineOpcode::VAddU32:
      emitLine(Twine("v_add_u32_e32 ") + def() + ", " + op(0) + ", " + op(1));
      return success();
    case MachineOpcode::VAndB32:
      emitLine(Twine("v_and_b32_e32 ") + def() + ", " + op(0) + ", " + op(1));
      return success();
    case MachineOpcode::VOrB32:
      emitLine(Twine("v_or_b32_e32 ") + def() + ", " + op(0) + ", " + op(1));
      return success();
    case MachineOpcode::VXorB32:
      emitLine(Twine("v_xor_b32_e32 ") + def() + ", " + op(0) + ", " + op(1));
      return success();
    case MachineOpcode::VLshlRevB32:
      emitLine(Twine("v_lshlrev_b32_e32 ") + def() + ", " + op(1) + ", " +
               op(0));
      return success();
    case MachineOpcode::VCmpEqU32:
      emitLine(Twine("v_cmp_eq_u32_e64 ") + def() + ", " + op(0) + ", " + op(1));
      return success();
    case MachineOpcode::VCmpNeU32:
      emitLine(Twine("v_cmp_ne_u32_e64 ") + def() + ", " + op(0) + ", " + op(1));
      return success();
    case MachineOpcode::VCmpLtU32:
      emitLine(Twine("v_cmp_lt_u32_e64 ") + def() + ", " + op(0) + ", " + op(1));
      return success();
    case MachineOpcode::VCmpLeU32:
      emitLine(Twine("v_cmp_le_u32_e64 ") + def() + ", " + op(0) + ", " + op(1));
      return success();
    case MachineOpcode::VCmpGtU32:
      emitLine(Twine("v_cmp_gt_u32_e64 ") + def() + ", " + op(0) + ", " + op(1));
      return success();
    case MachineOpcode::VCmpGeU32:
      emitLine(Twine("v_cmp_ge_u32_e64 ") + def() + ", " + op(0) + ", " + op(1));
      return success();
    case MachineOpcode::SMovB32:
      if (mi.defs.empty()) {
        if (op(0) != op(1))
          emitLine(Twine("s_mov_b32 ") + op(0) + ", " + op(1));
      } else {
        emitLine(Twine("s_mov_b32 ") + def() + ", " + op(0));
      }
      return success();
    case MachineOpcode::SAndSaveExecB32:
      emitLine(Twine("s_and_saveexec_b32 ") + def() + ", " + op(0));
      return success();
    case MachineOpcode::SCBranchExecZ:
      emitLine(Twine("s_cbranch_execz ") + op(0));
      return success();
    case MachineOpcode::SMovExecLo:
      emitLine(Twine("s_mov_b32 exec_lo, ") + op(0));
      return success();
    case MachineOpcode::VReadFirstLaneB32:
      emitLine(Twine("v_readfirstlane_b32 ") + def() + ", " + op(0));
      return success();
    case MachineOpcode::SSetPcB64:
      emitLine(StringRef("s_setpc_b64 s[30:31]"));
      return success();
    }
    llvm_unreachable("unknown machine opcode");
  }
};

} // namespace

LogicalResult mlir::wave::translateWaveToAMDGPU(Operation *op,
                                                raw_ostream &os) {
  return WaveAMDGPUEmitter(os).emit(op);
}

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
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/FormatVariadic.h"

using namespace mlir;
using namespace mlir::wave;

namespace {

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
  DenseMap<Value, std::string> values;
  unsigned nextVGPR = 0;
  unsigned nextSGPR = 0;
  unsigned nextLabel = 0;
  unsigned indent = 1;

  std::string makeVGPR() { return "v" + Twine(nextVGPR++).str(); }
  std::string makeSGPR() { return "s" + Twine(nextSGPR++).str(); }
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
    nextVGPR = 0;
    nextSGPR = 0;

    os << "\n\t.globl\t" << func.getSymName() << "\n";
    os << "\t.p2align\t8\n";
    os << "\t.type\t" << func.getSymName() << ",@function\n";
    os << func.getSymName() << ":\n";

    // MVP convention: integer arguments arrive in VGPRs. This keeps the direct
    // backend independent from the existing AMDGPU calling convention lowering.
    for (BlockArgument arg : func.getArguments())
      values[arg] = makeVGPR();

    if (!func.getBody().hasOneBlock())
      return func.emitError("wave AMDGPU backend only supports one-block funcs");
    for (Operation &op : func.getBody().front()) {
      if (failed(emitOperation(&op)))
        return failure();
    }

    os << "\t.size\t" << func.getSymName() << ", .-" << func.getSymName()
       << "\n";
    return success();
  }

  FailureOr<std::string> lookup(Value value) {
    auto it = values.find(value);
    if (it == values.end())
      return failure();
    return it->second;
  }

  std::string expect(Value value, Operation *user) {
    FailureOr<std::string> result = lookup(value);
    if (failed(result)) {
      user->emitError("value has no backend location");
      return "<invalid>";
    }
    return *result;
  }

  LogicalResult emitOperation(Operation *op) {
    if (auto constant = dyn_cast<arith::ConstantIntOp>(op))
      return emitConstant(constant);
    if (auto laneId = dyn_cast<LaneIdOp>(op))
      return emitLaneId(laneId);
    if (auto splat = dyn_cast<SplatOp>(op))
      return emitSplat(splat);
    if (auto binary = dyn_cast<BinaryOp>(op))
      return emitBinary(binary);
    if (auto cmp = dyn_cast<CmpIOp>(op))
      return emitCmp(cmp);
    if (auto ballot = dyn_cast<BallotOp>(op))
      return emitBallot(ballot);
    if (auto readFirst = dyn_cast<ReadFirstOp>(op))
      return emitReadFirst(readFirst);
    if (auto where = dyn_cast<WhereOp>(op))
      return emitWhere(where);
    if (auto store = dyn_cast<StoreOp>(op))
      return emitStore(store);
    if (auto ret = dyn_cast<func::ReturnOp>(op))
      return emitReturn(ret);
    if (isa<YieldOp>(op))
      return success();

    return op->emitError("unsupported operation in Wave AMDGPU backend");
  }

  LogicalResult emitConstant(arith::ConstantIntOp op) {
    values[op.getResult()] = Twine(op.value()).str();
    return success();
  }

  LogicalResult emitLaneId(LaneIdOp op) {
    auto simdType = cast<SimdType>(op.getType());
    if (!simdType.getElementType().isInteger(32) || simdType.getWidth() != 32)
      return op.emitError("backend supports only !wave.simd<i32, 32> lane_id");

    std::string dst = makeVGPR();
    emitLine(Twine("v_mbcnt_lo_u32_b32 ") + dst + ", -1, 0");
    values[op.getResult()] = dst;
    return success();
  }

  LogicalResult emitSplat(SplatOp op) {
    values[op.getResult()] = expect(op.getSource(), op);
    return success();
  }

  LogicalResult emitBinary(BinaryOp op) {
    std::string dst = makeVGPR();
    std::string lhs = expect(op.getLhs(), op);
    std::string rhs = expect(op.getRhs(), op);
    StringRef opcode =
        llvm::StringSwitch<StringRef>(op.getKind())
            .Case("addi", "v_add_u32_e32")
            .Case("andi", "v_and_b32_e32")
            .Case("ori", "v_or_b32_e32")
            .Case("xori", "v_xor_b32_e32")
            .Case("shli", "v_lshlrev_b32_e32")
            .Default("");
    if (opcode.empty())
      return op.emitError("unsupported wave.binary kind");
    if (op.getKind() == "shli")
      emitLine(Twine(opcode) + " " + dst + ", " + rhs + ", " + lhs);
    else
      emitLine(Twine(opcode) + " " + dst + ", " + lhs + ", " + rhs);
    values[op.getResult()] = dst;
    return success();
  }

  LogicalResult emitCmp(CmpIOp op) {
    auto maskType = cast<MaskType>(op.getType());
    if (maskType.getWidth() != 32)
      return op.emitError("backend supports only !wave.mask<32>");

    std::string dst = makeSGPR();
    std::string lhs = expect(op.getLhs(), op);
    std::string rhs = expect(op.getRhs(), op);
    StringRef cmp =
        llvm::StringSwitch<StringRef>(stringifyCmpIPredicate(op.getPredicate()))
            .Case("eq", "v_cmp_eq_u32_e64")
            .Case("ne", "v_cmp_ne_u32_e64")
            .Case("ult", "v_cmp_lt_u32_e64")
            .Case("ule", "v_cmp_le_u32_e64")
            .Case("ugt", "v_cmp_gt_u32_e64")
            .Case("uge", "v_cmp_ge_u32_e64")
            .Default("");
    if (cmp.empty())
      return op.emitError("unsupported wave.cmpi predicate");
    emitLine(Twine(cmp) + " " + dst + ", " + lhs + ", " + rhs);
    values[op.getResult()] = dst;
    return success();
  }

  LogicalResult emitBallot(BallotOp op) {
    std::string dst = makeSGPR();
    emitLine(Twine("s_mov_b32 ") + dst + ", " + expect(op.getMask(), op));
    values[op.getResult()] = dst;
    return success();
  }

  LogicalResult emitReadFirst(ReadFirstOp op) {
    std::string src = expect(op.getSource(), op);
    if (StringRef(src).starts_with("s")) {
      values[op.getResult()] = src;
      return success();
    }
    std::string dst = makeSGPR();
    emitLine(Twine("v_readfirstlane_b32 ") + dst + ", " + src);
    values[op.getResult()] = dst;
    return success();
  }

  LogicalResult emitStore(StoreOp op) {
    // This backend is intentionally text-only for now. Record where a real
    // memory backend would select a flat/global store.
    emitLine(Twine("; wave.store ") + expect(op.getValue(), op));
    return success();
  }

  LogicalResult emitWhere(WhereOp op) {
    std::string savedExec = makeSGPR();
    std::string endLabel = makeLabel("endif");
    emitLine(Twine("s_and_saveexec_b32 ") + savedExec + ", " +
             expect(op.getCondition(), op));
    emitLine(Twine("s_cbranch_execz ") + endLabel);
    if (failed(emitRegion(op.getThenRegion())))
      return failure();
    os << endLabel << ":\n";
    emitLine(Twine("s_mov_b32 exec_lo, ") + savedExec);
    if (!op.getElseRegion().empty()) {
      emitLine(StringRef("; otherwise region omitted in MVP backend"));
    }
    return success();
  }

  LogicalResult emitRegion(Region &region) {
    if (!region.hasOneBlock())
      return failure();
    for (Operation &op : region.front()) {
      if (failed(emitOperation(&op)))
        return failure();
    }
    return success();
  }

  LogicalResult emitReturn(func::ReturnOp op) {
    if (op.getNumOperands() > 1)
      return op.emitError("backend supports at most one return value");
    if (op.getNumOperands() == 1) {
      std::string ret = expect(op.getOperand(0), op);
      if (StringRef(ret).starts_with("v"))
        emitLine(Twine("v_readfirstlane_b32 s0, ") + ret);
      else
        emitLine(Twine("s_mov_b32 s0, ") + ret);
    }
    emitLine(StringRef("s_setpc_b64 s[30:31]"));
    return success();
  }
};

} // namespace

LogicalResult mlir::wave::translateWaveToAMDGPU(Operation *op,
                                                raw_ostream &os) {
  return WaveAMDGPUEmitter(os).emit(op);
}

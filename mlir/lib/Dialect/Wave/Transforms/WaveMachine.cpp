//===- WaveMachine.cpp - Wave to WaveMachine backend passes -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Wave/Transforms/Passes.h"

#include "Utils/AMDGPUBaseInfo.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Wave/IR/Wave.h"
#include "mlir/Dialect/WaveMachine/IR/WaveMachine.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/Debug.h"
#include "llvm/TargetParser/TargetParser.h"
#include <limits>
#include <optional>

namespace mlir::wave {
#define GEN_PASS_DEF_CONVERTWAVETOWAVEMACHINE
#define GEN_PASS_DEF_WAVEMACHINEABILOWERING
#define GEN_PASS_DEF_WAVEMACHINEHAZARDWAITS
#define GEN_PASS_DEF_WAVEMACHINEMETADATA
#define GEN_PASS_DEF_WAVEMACHINEREGALLOC
#define GEN_PASS_DEF_WAVEMACHINERESOURCEINFO
#include "mlir/Dialect/Wave/Transforms/Passes.h.inc"
} // namespace mlir::wave

using namespace mlir;
using namespace mlir::wave;

namespace {

enum class RegClass { SGPR, VGPR };

struct LiveInterval {
  Operation *def = nullptr;
  unsigned start = std::numeric_limits<unsigned>::max();
  unsigned end = 0;
};

static int64_t regClassCode(RegClass regClass) {
  return regClass == RegClass::SGPR ? 0 : 1;
}

static wavemachine::RegType getRegType(MLIRContext *ctx, RegClass regClass,
                                       unsigned width = 1) {
  return wavemachine::RegType::get(ctx, regClassCode(regClass), width);
}

static wavemachine::ImmType getImmType(MLIRContext *ctx) {
  return wavemachine::ImmType::get(ctx);
}

static bool isWaveMachineOp(Operation *op, StringRef name) {
  return op->getName().getStringRef() == ("wavemachine." + name).str();
}

static bool isWaveMachineOp(Operation *op) {
  return op->getName().getStringRef().starts_with("wavemachine.");
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

static Operation *createWMOp(OpBuilder &builder, Location loc, StringRef name,
                             ValueRange operands, Type resultType,
                             ArrayRef<NamedAttribute> attrs = {}) {
  SmallVector<Type, 1> resultTypes{resultType};
  return createWMOp(builder, loc, name, operands, resultTypes, attrs);
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

static Operation *createInstrNoResult(OpBuilder &builder, Location loc,
                                      StringRef name, ValueRange operands,
                                      ArrayRef<NamedAttribute> attrs = {}) {
  return createWMOp(builder, loc, name, operands, TypeRange{}, attrs);
}

static bool isReg(Value value) {
  return isa<wavemachine::RegType>(value.getType());
}

static bool isAllocatableReg(Value value) {
  return isReg(value);
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

static bool isSGPR(wavemachine::RegType type) { return type.getRegClass() == 0; }

static bool isVGPR(wavemachine::RegType type) { return type.getRegClass() == 1; }

class WaveMachineSelector {
public:
  explicit WaveMachineSelector(func::FuncOp func) : func(func), builder(func) {}

  LogicalResult run() {
    if (!func.getBody().hasOneBlock())
      return func.emitError("WaveMachine selection supports one-block funcs");

    Block &block = func.getBody().front();
    builder.setInsertionPointToStart(&block);
    for (auto [index, arg] : llvm::enumerate(func.getArguments())) {
      Type type = arg.getType();
      bool isMemref = isa<MemRefType>(type);
      RegClass regClass = isa<SimdType>(type) ? RegClass::VGPR : RegClass::SGPR;
      unsigned width = isMemref ? 2 : 1;
      Operation *argOp = createWMOp(
          builder, func.getLoc(), "arg", {}, getRegType(func.getContext(), regClass, width),
          {builder.getNamedAttr("index", builder.getI64IntegerAttr(index)),
           builder.getNamedAttr("memref", builder.getBoolAttr(isMemref))});
      values[arg] = argOp->getResult(0);
    }

    SmallVector<Operation *> topLevelOps;
    for (Operation &op : llvm::make_early_inc_range(block))
      if (!isWaveMachineOp(&op))
        topLevelOps.push_back(&op);

    for (Operation *op : topLevelOps) {
      if (failed(selectOperation(op)))
        return failure();
    }

    for (Operation *op : llvm::reverse(opsToErase))
      op->erase();

    auto oldType = func.getFunctionType();
    func.setType(FunctionType::get(func.getContext(), oldType.getInputs(),
                                   TypeRange{}));
    return success();
  }

private:
  func::FuncOp func;
  OpBuilder builder;
  DenseMap<Value, Value> values;
  SmallVector<Operation *> opsToErase;
  unsigned nextLabel = 0;

  std::string makeLabel(StringRef stem) {
    return (Twine(".Lwave_") + func.getSymName() + "_" + stem + "_" +
            Twine(nextLabel++))
        .str();
  }

  Value expect(Value value, Operation *user) {
    auto it = values.find(value);
    if (it != values.end())
      return it->second;
    user->emitError("value has no WaveMachine location");
    return createImm(builder, user->getLoc(), 0);
  }

  void eraseIfTopLevel(Operation *op) {
    if (op->getBlock()->getParentOp() == func)
      opsToErase.push_back(op);
  }

  LogicalResult selectOperation(Operation *op) {
    if (op->getBlock()->getParentOp() == func)
      builder.setInsertionPoint(op);
    if (auto constant = dyn_cast<arith::ConstantIntOp>(op))
      return selectConstant(constant);
    if (auto constant = dyn_cast<arith::ConstantIndexOp>(op))
      return selectConstantIndex(constant);
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
    if (auto fill = dyn_cast<FragmentFillOp>(op))
      return selectFragmentFill(fill);
    if (auto mma = dyn_cast<MmaOp>(op))
      return selectMma(mma);
    if (auto fragmentStore = dyn_cast<FragmentStoreOp>(op))
      return selectFragmentStore(fragmentStore);
    if (auto ret = dyn_cast<func::ReturnOp>(op))
      return selectReturn(ret);
    if (isa<YieldOp>(op))
      return success();

    return op->emitError("unsupported operation in WaveMachine selection");
  }

  LogicalResult selectConstant(arith::ConstantIntOp op) {
    values[op.getResult()] = createImm(builder, op.getLoc(), op.value());
    eraseIfTopLevel(op);
    return success();
  }

  LogicalResult selectConstantIndex(arith::ConstantIndexOp op) {
    values[op.getResult()] = createImm(builder, op.getLoc(), op.value());
    eraseIfTopLevel(op);
    return success();
  }

  LogicalResult selectLaneId(LaneIdOp op) {
    auto simdType = cast<SimdType>(op.getType());
    if (!simdType.getElementType().isInteger(32) || simdType.getWidth() != 32)
      return op.emitError("WaveMachine backend supports only !wave.simd<i32, 32> lane_id");
    values[op.getResult()] =
        createInstr(builder, op.getLoc(), "v_mbcnt_lo", {},
                    getRegType(op.getContext(), RegClass::VGPR));
    eraseIfTopLevel(op);
    return success();
  }

  LogicalResult selectSplat(SplatOp op) {
    values[op.getResult()] = expect(op.getSource(), op);
    eraseIfTopLevel(op);
    return success();
  }

  LogicalResult selectBinary(BinaryOp op) {
    StringRef machineOpcode =
        llvm::StringSwitch<StringRef>(op.getKind())
            .Case("addi", "v_add_u32")
            .Case("andi", "v_and_b32")
            .Case("ori", "v_or_b32")
            .Case("xori", "v_xor_b32")
            .Case("shli", "v_lshlrev_b32")
            .Default("");
    if (machineOpcode.empty())
      return op.emitError("unsupported wave.binary kind");
    values[op.getResult()] =
        createInstr(builder, op.getLoc(), machineOpcode,
                    {expect(op.getLhs(), op), expect(op.getRhs(), op)},
                    getRegType(op.getContext(), RegClass::VGPR));
    eraseIfTopLevel(op);
    return success();
  }

  LogicalResult selectCmp(CmpIOp op) {
    auto maskType = cast<MaskType>(op.getType());
    if (maskType.getWidth() != 32)
      return op.emitError("WaveMachine backend supports only !wave.mask<32>");

    StringRef machineOpcode =
        llvm::StringSwitch<StringRef>(stringifyCmpIPredicate(op.getPredicate()))
            .Case("eq", "v_cmp_eq_u32")
            .Case("ne", "v_cmp_ne_u32")
            .Case("ult", "v_cmp_lt_u32")
            .Case("ule", "v_cmp_le_u32")
            .Case("ugt", "v_cmp_gt_u32")
            .Case("uge", "v_cmp_ge_u32")
            .Default("");
    if (machineOpcode.empty())
      return op.emitError("unsupported wave.cmpi predicate");
    values[op.getResult()] =
        createInstr(builder, op.getLoc(), machineOpcode,
                    {expect(op.getLhs(), op), expect(op.getRhs(), op)},
                    getRegType(op.getContext(), RegClass::SGPR));
    eraseIfTopLevel(op);
    return success();
  }

  LogicalResult selectBallot(BallotOp op) {
    values[op.getResult()] = expect(op.getMask(), op);
    eraseIfTopLevel(op);
    return success();
  }

  LogicalResult selectReadFirst(ReadFirstOp op) {
    Value src = expect(op.getSource(), op);
    if (auto regType = dyn_cast<wavemachine::RegType>(src.getType());
        regType && regType.getRegClass() == regClassCode(RegClass::SGPR)) {
      values[op.getResult()] = src;
      eraseIfTopLevel(op);
      return success();
    }
    values[op.getResult()] =
        createInstr(builder, op.getLoc(), "v_readfirstlane_b32", src,
                    getRegType(op.getContext(), RegClass::SGPR));
    eraseIfTopLevel(op);
    return success();
  }

  LogicalResult selectStore(StoreOp op) {
    if (op.getIndices().size() != 1)
      return op.emitError("WaveMachine backend expects exactly one store index");
    Value index = expect(op.getIndices().front(), op);
    Value byteOffset =
        createInstr(builder, op.getLoc(), "v_lshlrev_b32",
                    {index, createImm(builder, op.getLoc(), 2)},
                    getRegType(op.getContext(), RegClass::VGPR));
    createInstrNoResult(builder, op.getLoc(), "global_store_b32",
                        {byteOffset, expect(op.getValue(), op),
                         expect(op.getMemref(), op)});
    eraseIfTopLevel(op);
    return success();
  }

  LogicalResult selectFragmentFill(FragmentFillOp op) {
    auto fragmentType = cast<FragmentType>(op.getResult().getType());
    Value source = expect(op.getSource(), op);
    values[op.getResult()] =
        createInstr(builder, op.getLoc(), "v_mov_b32_tuple", source,
                    getRegType(op.getContext(), RegClass::VGPR,
                               fragmentType.getRegisters()),
                    {builder.getNamedAttr(
                        "registers",
                        builder.getI64IntegerAttr(fragmentType.getRegisters()))});
    eraseIfTopLevel(op);
    return success();
  }

  LogicalResult selectMma(MmaOp op) {
    if (op.getKind() != "wmma.i32.16x16x16.iu8" &&
        op.getKind() != "wmma.f32.16x16x16.f16")
      return op.emitError("unsupported WaveMachine matrix operation kind");
    auto resultType = cast<FragmentType>(op.getResult().getType());
    StringRef machineOpcode =
        op.getKind() == "wmma.i32.16x16x16.iu8"
            ? "wmma_i32_16x16x16_iu8"
            : "wmma_f32_16x16x16_f16";
    values[op.getResult()] =
        createInstr(builder, op.getLoc(), machineOpcode,
                    {expect(op.getA(), op), expect(op.getB(), op),
                     expect(op.getAcc(), op)},
                    getRegType(op.getContext(), RegClass::VGPR,
                               resultType.getRegisters()));
    eraseIfTopLevel(op);
    return success();
  }

  LogicalResult selectFragmentStore(FragmentStoreOp op) {
    auto fragmentType = cast<FragmentType>(op.getFragment().getType());
    if (op.getIndices().size() != 1)
      return op.emitError("WaveMachine backend expects one fragment store index");

    Value lane = createInstr(builder, op.getLoc(), "v_mbcnt_lo", {},
                             getRegType(op.getContext(), RegClass::VGPR));
    Value byteOffset =
        createInstr(builder, op.getLoc(), "v_lshlrev_b32",
                    {lane, createImm(builder, op.getLoc(), 5)},
                    getRegType(op.getContext(), RegClass::VGPR));

    Value baseIndex = expect(op.getIndices().front(), op);
    if (auto baseDef = baseIndex.getDefiningOp();
        baseDef && isWaveMachineOp(baseDef, "imm")) {
      int64_t base = baseDef->getAttrOfType<IntegerAttr>("value").getInt();
      if (base != 0)
        return op.emitError(
            "WaveMachine backend expects a zero fragment store base index");
    } else {
      return op.emitError("WaveMachine backend expects a constant fragment store base index");
    }

    for (int64_t component = 0, e = fragmentType.getRegisters(); component != e;
         ++component) {
      createInstrNoResult(
          builder, op.getLoc(), "global_store_tuple_b32",
          {byteOffset, expect(op.getFragment(), op), expect(op.getMemref(), op)},
          {builder.getNamedAttr("component",
                                builder.getI64IntegerAttr(component))});
    }
    eraseIfTopLevel(op);
    return success();
  }

  LogicalResult selectWhere(WhereOp op) {
    std::string endLabel = makeLabel("endif");
    std::string elseLabel = op.getElseRegion().empty() ? endLabel : makeLabel("else");
    Value condition = expect(op.getCondition(), op);
    Value savedExec =
        createInstr(builder, op.getLoc(), "s_and_saveexec_b32", condition,
                    getRegType(op.getContext(), RegClass::SGPR));
    createInstrNoResult(builder, op.getLoc(), "s_cbranch_execz", {},
                        {builder.getNamedAttr("label", builder.getStringAttr(elseLabel))});
    if (failed(selectRegion(op.getThenRegion())))
      return failure();
    if (!op.getElseRegion().empty()) {
      createInstrNoResult(builder, op.getLoc(), "s_andn2_exec_b32",
                          {savedExec, condition});
      createInstrNoResult(builder, op.getLoc(), "s_cbranch_execz", {},
                          {builder.getNamedAttr("label", builder.getStringAttr(endLabel))});
      createInstrNoResult(builder, op.getLoc(), "label", {},
                          {builder.getNamedAttr("name", builder.getStringAttr(elseLabel))});
      if (failed(selectRegion(op.getElseRegion())))
        return failure();
    }
    createInstrNoResult(builder, op.getLoc(), "label", {},
                        {builder.getNamedAttr("name", builder.getStringAttr(endLabel))});
    createInstrNoResult(builder, op.getLoc(), "s_mov_exec_lo", savedExec);
    eraseIfTopLevel(op);
    return success();
  }

  LogicalResult selectRegion(Region &region) {
    if (!region.hasOneBlock())
      return failure();
    for (Operation &op : llvm::make_early_inc_range(region.front())) {
      if (failed(selectOperation(&op)))
        return failure();
    }
    return success();
  }

  LogicalResult selectReturn(func::ReturnOp op) {
    if (op.getNumOperands() > 1)
      return op.emitError("WaveMachine backend supports at most one return value");
    if (func->hasAttr("wave.kernel")) {
      if (op.getNumOperands() != 0)
        return op.emitError("kernel functions must return void");
      createInstrNoResult(builder, op.getLoc(), "s_endpgm", {});
      op.getOperandsMutable().clear();
      return success();
    }

    if (op.getNumOperands() == 1) {
      Value ret = expect(op.getOperand(0), op);
      auto regType = dyn_cast<wavemachine::RegType>(ret.getType());
      if (regType && regType.getRegClass() == regClassCode(RegClass::VGPR))
        ret = createInstr(builder, op.getLoc(), "v_readfirstlane_b32", ret,
                          getRegType(op.getContext(), RegClass::SGPR));
      createInstrNoResult(builder, op.getLoc(), "s_mov_b32", ret,
                          {builder.getNamedAttr("dst", builder.getStringAttr("s0"))});
    }
    createInstrNoResult(builder, op.getLoc(), "s_setpc_b64", {});
    op.getOperandsMutable().clear();
    return success();
  }
};

struct ConvertWaveToWaveMachinePass
    : public wave::impl::ConvertWaveToWaveMachineBase<
          ConvertWaveToWaveMachinePass> {
  void runOnOperation() override {
    ModuleOp module = getOperation();
    for (func::FuncOp func : module.getOps<func::FuncOp>()) {
      if (failed(WaveMachineSelector(func).run()))
        return signalPassFailure();
    }
  }
};

struct WaveMachineABILoweringPass
    : public wave::impl::WaveMachineABILoweringBase<
          WaveMachineABILoweringPass> {
  void runOnOperation() override {
    ModuleOp module = getOperation();
    OpBuilder builder(module.getContext());
    for (func::FuncOp func : module.getOps<func::FuncOp>()) {
      if (!func->hasAttr("wave.kernel"))
        continue;
      Block &block = func.getBody().front();

      unsigned offset = 0;
      for (Operation &op : llvm::make_early_inc_range(block)) {
        if (!isWaveMachineOp(&op, "arg"))
          continue;
        if (op.getNumResults() != 1) {
          op.emitError("wavemachine-abi-lowering expects wavemachine.arg "
                       "to have one result");
          return signalPassFailure();
        }
        auto regType = dyn_cast<wavemachine::RegType>(op.getResult(0).getType());
        if (!regType || !isSGPR(regType)) {
          op.emitError("wavemachine-abi-lowering expects kernel arguments "
                       "to be SGPR WaveMachine registers");
          return signalPassFailure();
        }
        auto memrefAttr = op.getAttrOfType<BoolAttr>("memref");
        if (!memrefAttr) {
          op.emitError("wavemachine-abi-lowering expects wavemachine.arg "
                       "to have a memref attribute");
          return signalPassFailure();
        }
        bool isMemref = false;
        isMemref = memrefAttr.getValue();
        if ((isMemref && regType.getWidth() != 2) ||
            (!isMemref && regType.getWidth() != 1)) {
          op.emitError("wavemachine-abi-lowering found argument register "
                       "width inconsistent with memref attribute");
          return signalPassFailure();
        }
        unsigned size = isMemref ? 8 : 4;

        builder.setInsertionPoint(&op);
        Value offsetImm = createImm(builder, op.getLoc(), offset);
        Value loaded =
            createInstr(builder, op.getLoc(), isMemref ? "s_load_b64" : "s_load_b32",
                        offsetImm, op.getResult(0).getType(),
                        {builder.getNamedAttr("base",
                                              builder.getStringAttr("s[0:1]"))});
        op.getResult(0).replaceAllUsesWith(loaded);
        op.erase();
        offset += size;
      }
      unsigned kernargSize = (std::max(offset, 4u) + 7u) & ~7u;
      func->setAttr("wavemachine.kernarg_size",
                    builder.getI64IntegerAttr(kernargSize));
    }
  }
};

struct WaveMachineHazardWaitsPass
    : public wave::impl::WaveMachineHazardWaitsBase<WaveMachineHazardWaitsPass> {
  void runOnOperation() override {
    ModuleOp module = getOperation();
    OpBuilder builder(module.getContext());
    for (func::FuncOp func : module.getOps<func::FuncOp>()) {
      bool pendingLgkmWait = false;
      for (Operation &op : llvm::make_early_inc_range(func.getBody().front())) {
        if (!isWaveMachineOp(&op))
          continue;
        if (func->hasAttr("wave.kernel") && isWaveMachineOp(&op, "arg")) {
          op.emitError("wavemachine-insert-hazard-waits expects "
                       "ABI-lowered kernel arguments");
          return signalPassFailure();
        }
        if (isSMEMLoad(&op) && !op.getAttrOfType<StringAttr>("base")) {
          op.emitError("wavemachine-insert-hazard-waits expects scalar "
                       "memory loads to carry a base register attribute");
          return signalPassFailure();
        }

        if (isVALU(&op) && pendingLgkmWait) {
          builder.setInsertionPoint(&op);
          createInstrNoResult(builder, op.getLoc(), "s_delay_alu",
                              createImm(builder, op.getLoc(), 1));
          pendingLgkmWait = false;
        }

        if (isWaveMachineOp(&op, "s_waitcnt")) {
          auto imm = getImmediate(op.getOperand(0));
          if (!imm)
            continue;
          llvm::AMDGPU::IsaVersion gfx11{11, 0, 0};
          unsigned vm = 0;
          unsigned exp = 0;
          unsigned lg = 0;
          llvm::AMDGPU::decodeWaitcnt(gfx11, *imm, vm, exp, lg);
          pendingLgkmWait = lg != llvm::AMDGPU::decodeLgkmcnt(
                                      gfx11, llvm::AMDGPU::getWaitcntBitMask(gfx11));
        }
      }
    }
  }

  std::optional<unsigned> getImmediate(Value value) {
    Operation *def = value.getDefiningOp();
    if (!def || !isWaveMachineOp(def, "imm"))
      return std::nullopt;
    return static_cast<unsigned>(
        def->getAttrOfType<IntegerAttr>("value").getInt());
  }
};

struct WaveMachineRegAllocPass
    : public wave::impl::WaveMachineRegAllocBase<WaveMachineRegAllocPass> {
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
        if (!isAllocatableReg(result))
          continue;
        auto regType = cast<wavemachine::RegType>(result.getType());
        if (!isSGPR(regType) && !isVGPR(regType))
          return op->emitError("wavemachine-reg-alloc supports only SGPR(0) "
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

  LogicalResult allocateClass(func::FuncOp func, MutableArrayRef<LiveInterval> intervals,
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

struct WaveMachineResourceInfoPass
    : public wave::impl::WaveMachineResourceInfoBase<
          WaveMachineResourceInfoPass> {
  void runOnOperation() override {
    OpBuilder builder(getOperation().getContext());
    for (func::FuncOp func : getOperation().getOps<func::FuncOp>()) {
      unsigned maxSGPR = func->hasAttr("wave.kernel") ? 2 : 0;
      unsigned maxVGPR = 0;
      for (Operation &op : func.getBody().front()) {
        if (op.getNumResults() == 0)
          continue;
        auto regType = dyn_cast<wavemachine::RegType>(op.getResult(0).getType());
        if (!regType)
          continue;
        auto physAttr = op.getAttrOfType<IntegerAttr>("phys");
        if (!physAttr) {
          op.emitError("wavemachine-resource-info requires allocated register "
                       "results");
          return signalPassFailure();
        }
        unsigned end = physAttr.getInt() + regType.getWidth();
        if (isSGPR(regType))
          maxSGPR = std::max(maxSGPR, end);
        if (isVGPR(regType))
          maxVGPR = std::max(maxVGPR, end);
      }
      func->setAttr("wavemachine.sgpr_count",
                    builder.getI64IntegerAttr(std::max(maxSGPR, func->hasAttr("wave.kernel") ? 6u : 1u)));
      func->setAttr("wavemachine.vgpr_count",
                    builder.getI64IntegerAttr(std::max(maxVGPR, 1u)));
    }
  }
};

struct WaveMachineMetadataPass
    : public wave::impl::WaveMachineMetadataBase<WaveMachineMetadataPass> {
  void runOnOperation() override {
    OpBuilder builder(getOperation().getContext());
    getOperation()->setAttr("wavemachine.target",
                            builder.getStringAttr("amdgcn-amd-amdhsa--gfx1100"));
    for (func::FuncOp func : getOperation().getOps<func::FuncOp>()) {
      if (!func->hasAttr("wave.kernel"))
        continue;
      if (!func->hasAttr("wavemachine.kernarg_size") ||
          !func->hasAttr("wavemachine.sgpr_count") ||
          !func->hasAttr("wavemachine.vgpr_count")) {
        func.emitError("wavemachine-metadata requires ABI and resource "
                       "attributes on kernels");
        return signalPassFailure();
      }
      func->setAttr("wavemachine.metadata", builder.getUnitAttr());
    }
  }
};

} // namespace

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
#include "mlir/Dialect/Wave/IR/Wave.h"
#include "mlir/Dialect/Wave/IR/WaveAMD.h"
#include "mlir/Dialect/WaveMachine/IR/WaveMachine.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/Debug.h"
#include "llvm/TargetParser/TargetParser.h"
#include <limits>
#include <optional>

namespace mlir::wave {
#define GEN_PASS_DEF_CONVERTWAVEAMDTOWAVEMACHINE
#include "mlir/Dialect/Wave/Transforms/Passes.h.inc"
} // namespace mlir::wave

using namespace mlir;
using namespace mlir::wave;
using namespace mlir::waveamd;

namespace {

struct LiveInterval {
  Operation *def = nullptr;
  unsigned start = std::numeric_limits<unsigned>::max();
  unsigned end = 0;
};

static wavemachine::RegType getRegType(MLIRContext *ctx,
                                       wavemachine::RegClass regClass,
                                       unsigned width = 1) {
  return wavemachine::RegType::get(ctx, regClass, width, -1);
}

static wavemachine::ImmType getImmType(MLIRContext *ctx) {
  return wavemachine::ImmType::get(ctx);
}

static wavemachine::MemTokenType getMemTokenType(MLIRContext *ctx) {
  return wavemachine::MemTokenType::get(ctx);
}

static bool isWaveMachineOp(Operation *op) {
  return op->getName().getDialectNamespace() ==
         wavemachine::WaveMachineDialect::getDialectNamespace();
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
      bool isPtr = isa<PtrType>(type);
      wavemachine::RegClass regClass =
          isa<SimdType>(type) ? wavemachine::RegClass::VGPR
                              : wavemachine::RegClass::SGPR;
      unsigned width = isPtr ? pointerBaseWidth(type) : 1;
      Operation *argOp = createWMOp(
          builder, func.getLoc(), "arg", {}, getRegType(func.getContext(), regClass, width),
          {builder.getNamedAttr("index", builder.getI64IntegerAttr(index)),
           builder.getNamedAttr("pointer", builder.getBoolAttr(isPtr))});
      values[arg] = argOp->getResult(0);
      if (isPtr) {
        pointerBases[arg] = argOp->getResult(0);
        pointerOffsets[arg] = createImm(builder, func.getLoc(), 0);
        pointerBuffers[arg] = isBufferPointer(type);
      }
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
  DenseMap<Value, Value> pointerBases;
  DenseMap<Value, Value> pointerOffsets;
  DenseMap<Value, bool> pointerBuffers;
  SmallVector<Operation *> opsToErase;
  unsigned nextLabel = 0;

  bool isBufferPointer(Type type) {
    if (auto simd = dyn_cast<SimdType>(type))
      type = simd.getElementType();
    auto ptr = dyn_cast<PtrType>(type);
    return ptr && isa<waveamd::BufferAddressSpaceAttr>(ptr.getAddressSpace());
  }

  unsigned pointerBaseWidth(Type type) {
    return isBufferPointer(type) ? 4 : 2;
  }

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
    if (auto ptrAdd = dyn_cast<PtrAddOp>(op))
      return selectPtrAdd(ptrAdd);
    if (auto makeBuffer = dyn_cast<waveamd::MakeBufferOp>(op))
      return selectMakeBuffer(makeBuffer);
    if (auto token = dyn_cast<TokenOp>(op))
      return selectToken(token);
    if (auto after = dyn_cast<AfterOp>(op))
      return selectTokenJoin(after);
    if (auto join = dyn_cast<JoinOp>(op))
      return selectTokenJoin(join);
    if (auto wait = dyn_cast<WaitOp>(op))
      return selectWait(wait);
    if (auto where = dyn_cast<WhereOp>(op))
      return selectWhere(where);
    if (auto store = dyn_cast<StoreOp>(op))
      return selectStore(store);
    if (auto fill = dyn_cast<waveamd::FragmentFillOp>(op))
      return selectFragmentFill(fill);
    if (auto mma = dyn_cast<waveamd::MmaOp>(op))
      return selectMma(mma);
    if (auto fragmentStore = dyn_cast<waveamd::FragmentStoreOp>(op))
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
                    getRegType(op.getContext(), wavemachine::RegClass::VGPR));
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
                    getRegType(op.getContext(), wavemachine::RegClass::VGPR));
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
                    getRegType(op.getContext(), wavemachine::RegClass::SGPR));
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
        regType && regType.getRegClass() == wavemachine::RegClass::SGPR) {
      values[op.getResult()] = src;
      eraseIfTopLevel(op);
      return success();
    }
    values[op.getResult()] =
        createInstr(builder, op.getLoc(), "v_readfirstlane_b32", src,
                    getRegType(op.getContext(), wavemachine::RegClass::SGPR));
    eraseIfTopLevel(op);
    return success();
  }

  LogicalResult selectStore(StoreOp op) {
    auto baseIt = pointerBases.find(op.getPtr());
    auto offsetIt = pointerOffsets.find(op.getPtr());
    auto bufferIt = pointerBuffers.find(op.getPtr());
    if (baseIt == pointerBases.end() || offsetIt == pointerOffsets.end())
      return op.emitError("WaveMachine backend expects selected wave pointer");
    SmallVector<Value> operands{offsetIt->second, expect(op.getValue(), op),
                                baseIt->second};
    if (Value dependency = op.getDependency())
      operands.push_back(expect(dependency, op));
    Operation *store = createWMOp(
        builder, op.getLoc(),
        bufferIt != pointerBuffers.end() && bufferIt->second
            ? "buffer_store_b32"
            : "global_store_b32",
        operands, getMemTokenType(op.getContext()));
    values[op.getToken()] = store->getResult(0);
    eraseIfTopLevel(op);
    return success();
  }

  unsigned elementSizeBytes(Type type) {
    if (auto ptr = dyn_cast<PtrType>(type))
      type = ptr.getElementType();
    if (auto simd = dyn_cast<SimdType>(type))
      type = cast<PtrType>(simd.getElementType()).getElementType();
    if (type.isInteger(8))
      return 1;
    if (type.isInteger(16) || type.isF16())
      return 2;
    if (type.isIntOrFloat() && type.getIntOrFloatBitWidth() == 32)
      return 4;
    return 4;
  }

  std::optional<int64_t> getImmediateValue(Value value) {
    auto imm = value.getDefiningOp<wavemachine::ImmOp>();
    if (!imm)
      return std::nullopt;
    return imm->getAttrOfType<IntegerAttr>("value").getInt();
  }

  Value addByteOffsets(Location loc, Value lhs, Value rhs) {
    std::optional<int64_t> lhsImm = getImmediateValue(lhs);
    std::optional<int64_t> rhsImm = getImmediateValue(rhs);
    if (lhsImm && rhsImm)
      return createImm(builder, loc, *lhsImm + *rhsImm);
    if (lhsImm && *lhsImm == 0)
      return rhs;
    if (rhsImm && *rhsImm == 0)
      return lhs;
    return createInstr(builder, loc, "v_add_u32", {lhs, rhs},
                       getRegType(builder.getContext(), wavemachine::RegClass::VGPR));
  }

  LogicalResult selectPtrAdd(PtrAddOp op) {
    auto baseIt = pointerBases.find(op.getBase());
    auto offsetIt = pointerOffsets.find(op.getBase());
    if (baseIt == pointerBases.end() || offsetIt == pointerOffsets.end())
      return op.emitError("WaveMachine backend expects selected base pointer");

    Value offset = expect(op.getOffset(), op);
    unsigned size = elementSizeBytes(op.getBase().getType());
    Value byteOffset = offset;
    if (auto offsetDef = offset.getDefiningOp<wavemachine::ImmOp>()) {
      int64_t scaled =
          offsetDef->getAttrOfType<IntegerAttr>("value").getInt() * size;
      byteOffset = createImm(builder, op.getLoc(), scaled);
    } else if (size != 1) {
      byteOffset = createInstr(builder, op.getLoc(), "v_lshlrev_b32",
                               {offset, createImm(builder, op.getLoc(),
                                                  llvm::Log2_32(size))},
                               getRegType(op.getContext(), wavemachine::RegClass::VGPR));
    }
    byteOffset = addByteOffsets(op.getLoc(), offsetIt->second, byteOffset);

    pointerBases[op.getResult()] = baseIt->second;
    pointerOffsets[op.getResult()] = byteOffset;
    pointerBuffers[op.getResult()] = pointerBuffers.lookup(op.getBase());
    values[op.getResult()] = baseIt->second;
    eraseIfTopLevel(op);
    return success();
  }

  LogicalResult selectMakeBuffer(waveamd::MakeBufferOp op) {
    auto baseIt = pointerBases.find(op.getBase());
    auto offsetIt = pointerOffsets.find(op.getBase());
    if (baseIt == pointerBases.end() || offsetIt == pointerOffsets.end())
      return op.emitError("WaveMachine backend expects selected base pointer");
    Operation *descriptor = createWMOp(
        builder, op.getLoc(), "make_buffer_rsrc",
        {baseIt->second, expect(op.getRange(), op)},
        getRegType(op.getContext(), wavemachine::RegClass::SGPR, 4));
    pointerBases[op.getResult()] = descriptor->getResult(0);
    pointerOffsets[op.getResult()] = offsetIt->second;
    pointerBuffers[op.getResult()] = true;
    values[op.getResult()] = descriptor->getResult(0);
    eraseIfTopLevel(op);
    return success();
  }

  LogicalResult selectToken(TokenOp op) {
    Operation *token = createWMOp(builder, op.getLoc(), "token", {},
                                  getMemTokenType(op.getContext()));
    values[op.getResult()] = token->getResult(0);
    eraseIfTopLevel(op);
    return success();
  }

  LogicalResult selectTokenJoin(Operation *op) {
    SmallVector<Value> operands;
    for (Value dependency : op->getOperands())
      operands.push_back(expect(dependency, op));
    Operation *join = createWMOp(builder, op->getLoc(), "token_join", operands,
                                 getMemTokenType(op->getContext()));
    values[op->getResult(0)] = join->getResult(0);
    eraseIfTopLevel(op);
    return success();
  }

  LogicalResult selectWait(WaitOp op) {
    SmallVector<Value> operands;
    for (Value dependency : op.getDependencies())
      operands.push_back(expect(dependency, op));
    createInstrNoResult(builder, op.getLoc(), "wait", operands);
    eraseIfTopLevel(op);
    return success();
  }

  LogicalResult selectFragmentFill(waveamd::FragmentFillOp op) {
    auto fragmentType = cast<waveamd::FragmentType>(op.getResult().getType());
    Value source = expect(op.getSource(), op);
    values[op.getResult()] =
        createInstr(builder, op.getLoc(), "v_mov_b32_tuple", source,
                    getRegType(op.getContext(), wavemachine::RegClass::VGPR,
                               fragmentType.getRegisters()),
                    {builder.getNamedAttr(
                        "registers",
                        builder.getI64IntegerAttr(fragmentType.getRegisters()))});
    eraseIfTopLevel(op);
    return success();
  }

  LogicalResult selectMma(waveamd::MmaOp op) {
    if (op.getKind() != "wmma.i32.16x16x16.iu8" &&
        op.getKind() != "wmma.f32.16x16x16.f16")
      return op.emitError("unsupported WaveMachine matrix operation kind");
    auto resultType = cast<waveamd::FragmentType>(op.getResult().getType());
    StringRef machineOpcode =
        op.getKind() == "wmma.i32.16x16x16.iu8"
            ? "wmma_i32_16x16x16_iu8"
            : "wmma_f32_16x16x16_f16";
    values[op.getResult()] =
        createInstr(builder, op.getLoc(), machineOpcode,
                    {expect(op.getA(), op), expect(op.getB(), op),
                     expect(op.getAcc(), op)},
                    getRegType(op.getContext(), wavemachine::RegClass::VGPR,
                               resultType.getRegisters()));
    eraseIfTopLevel(op);
    return success();
  }

  LogicalResult selectFragmentStore(waveamd::FragmentStoreOp op) {
    auto fragmentType = cast<waveamd::FragmentType>(op.getFragment().getType());
    auto baseIt = pointerBases.find(op.getPtr());
    auto offsetIt = pointerOffsets.find(op.getPtr());
    if (baseIt == pointerBases.end() || offsetIt == pointerOffsets.end())
      return op.emitError("WaveMachine backend expects selected wave pointer");
    Value lane = createInstr(builder, op.getLoc(), "v_mbcnt_lo", {},
                             getRegType(op.getContext(), wavemachine::RegClass::VGPR));
    Value byteOffset =
        createInstr(builder, op.getLoc(), "v_lshlrev_b32",
                    {lane, createImm(builder, op.getLoc(), 5)},
                    getRegType(op.getContext(), wavemachine::RegClass::VGPR));
    byteOffset = addByteOffsets(op.getLoc(), offsetIt->second, byteOffset);

    SmallVector<Value> storeTokens;
    for (int64_t component = 0, e = fragmentType.getRegisters(); component != e;
         ++component) {
      SmallVector<Value> operands{byteOffset, expect(op.getFragment(), op),
                                  baseIt->second};
      if (Value dependency = op.getDependency())
        operands.push_back(expect(dependency, op));
      Operation *store = createWMOp(
          builder, op.getLoc(), "global_store_tuple_b32", operands,
          getMemTokenType(op.getContext()),
          {builder.getNamedAttr("component",
                                builder.getI64IntegerAttr(component))});
      storeTokens.push_back(store->getResult(0));
    }
    Operation *token = createWMOp(builder, op.getLoc(), "token_join", storeTokens,
                                 getMemTokenType(op.getContext()));
    values[op.getToken()] = token->getResult(0);
    eraseIfTopLevel(op);
    return success();
  }

  LogicalResult selectWhere(WhereOp op) {
    std::string endLabel = makeLabel("endif");
    std::string elseLabel = op.getElseRegion().empty() ? endLabel : makeLabel("else");
    Value condition = expect(op.getCondition(), op);
    Value savedExec =
        createInstr(builder, op.getLoc(), "s_and_saveexec_b32", condition,
                    getRegType(op.getContext(), wavemachine::RegClass::SGPR));
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
      if (regType && regType.getRegClass() == wavemachine::RegClass::VGPR)
        ret = createInstr(builder, op.getLoc(), "v_readfirstlane_b32", ret,
                          getRegType(op.getContext(), wavemachine::RegClass::SGPR));
      createInstrNoResult(builder, op.getLoc(), "s_mov_b32", ret,
                          {builder.getNamedAttr("dst", builder.getStringAttr("s0"))});
    }
    createInstrNoResult(builder, op.getLoc(), "s_setpc_b64", {});
    op.getOperandsMutable().clear();
    return success();
  }
};

struct ConvertWaveAMDToWaveMachinePass
    : public wave::impl::ConvertWaveAMDToWaveMachineBase<
          ConvertWaveAMDToWaveMachinePass> {
  void runOnOperation() override {
    ModuleOp module = getOperation();
    for (func::FuncOp func : module.getOps<func::FuncOp>()) {
      if (failed(WaveMachineSelector(func).run()))
        return signalPassFailure();
    }
  }
};

} // namespace

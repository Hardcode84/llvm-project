//===- WaveAMD.cpp - WaveAMD dialect ----------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Wave/IR/WaveAMD.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::waveamd;

#include "mlir/Dialect/Wave/IR/WaveAMDOpsDialect.cpp.inc"

void WaveAMDDialect::initialize() {
  registerAttributes();
  registerTypes();
  addOperations<
#define GET_OP_LIST
#include "mlir/Dialect/Wave/IR/WaveAMDOps.cpp.inc"
      >();
}

void WaveAMDDialect::registerAttributes() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "mlir/Dialect/Wave/IR/WaveAMDOpsAttributes.cpp.inc"
      >();
}

void WaveAMDDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "mlir/Dialect/Wave/IR/WaveAMDOpsTypes.cpp.inc"
      >();
}

LogicalResult MakeBufferOp::verify() {
  auto baseType = cast<wave::PtrType>(getBase().getType());
  auto resultType = cast<wave::PtrType>(getResult().getType());
  if (!isa<wave::GlobalAddressSpaceAttr>(baseType.getAddressSpace()))
    return emitOpError("base must be a global wave pointer");
  if (!isa<BufferAddressSpaceAttr>(resultType.getAddressSpace()))
    return emitOpError("result must be a waveamd buffer pointer");
  if (baseType.getElementType() != resultType.getElementType())
    return emitOpError("base and result element types must match");
  if (!getRange().getType().isInteger(32))
    return emitOpError("range must be i32 bytes");
  return success();
}

LogicalResult FragmentFillOp::verify() {
  auto fragmentType = cast<FragmentType>(getResult().getType());
  if (!getSource().getType().isInteger(32))
    return emitOpError("source must be an i32 bit pattern");
  if (fragmentType.getWaveSize() != 32)
    return emitOpError("only wave32 fragments are supported for now");
  if (fragmentType.getRole() != 0 && fragmentType.getRole() != 1 &&
      fragmentType.getRole() != 2)
    return emitOpError("fragment role must be 0 (A), 1 (B), or 2 (acc)");
  if (fragmentType.getRows() != 16 || fragmentType.getColumns() != 16)
    return emitOpError("only 16x16 fragments are supported for now");
  if (fragmentType.getRole() == 0 || fragmentType.getRole() == 1) {
    bool isIU8 = fragmentType.getElementType().isInteger(8) &&
                 fragmentType.getRegisters() == 4;
    bool isF16 = fragmentType.getElementType().isF16() &&
                 fragmentType.getRegisters() == 8;
    if (!isIU8 && !isF16)
      return emitOpError("A/B fragments must be i8 fragments with 4 registers "
                         "or f16 fragments with 8 registers");
  }
  if (fragmentType.getRole() == 2 &&
      (!fragmentType.getElementType().isIntOrFloat() ||
       fragmentType.getElementType().getIntOrFloatBitWidth() != 32 ||
       fragmentType.getRegisters() != 8))
    return emitOpError("accumulator fragments must be 32-bit fragments with 8 registers");
  return success();
}

LogicalResult MmaOp::verify() {
  if (getKind() != "wmma.i32.16x16x16.iu8" &&
      getKind() != "wmma.f32.16x16x16.f16")
    return emitOpError("unsupported matrix operation kind");

  auto aType = cast<FragmentType>(getA().getType());
  auto bType = cast<FragmentType>(getB().getType());
  auto accType = cast<FragmentType>(getAcc().getType());
  auto resultType = cast<FragmentType>(getResult().getType());

  auto isIU8AB = [](FragmentType type, int64_t role) {
    return type.getRole() == role && type.getElementType().isInteger(8) &&
           type.getRows() == 16 && type.getColumns() == 16 &&
           type.getWaveSize() == 32 && type.getRegisters() == 4;
  };
  auto isI32Acc = [](FragmentType type) {
    return type.getRole() == 2 && type.getElementType().isInteger(32) &&
           type.getRows() == 16 && type.getColumns() == 16 &&
           type.getWaveSize() == 32 && type.getRegisters() == 8;
  };

  auto isF16AB = [](FragmentType type, int64_t role) {
    return type.getRole() == role && type.getElementType().isF16() &&
           type.getRows() == 16 && type.getColumns() == 16 &&
           type.getWaveSize() == 32 && type.getRegisters() == 8;
  };
  auto isF32Acc = [](FragmentType type) {
    return type.getRole() == 2 && type.getElementType().isF32() &&
           type.getRows() == 16 && type.getColumns() == 16 &&
           type.getWaveSize() == 32 && type.getRegisters() == 8;
  };

  if (getKind() == "wmma.i32.16x16x16.iu8") {
    if (!isIU8AB(aType, 0))
      return emitOpError("A operand must be a 16x16 i8 wave32 fragment with 4 registers");
    if (!isIU8AB(bType, 1))
      return emitOpError("B operand must be a 16x16 i8 wave32 fragment with 4 registers");
    if (!isI32Acc(accType))
      return emitOpError("accumulator must be a 16x16 i32 wave32 fragment with 8 registers");
  } else {
    if (!isF16AB(aType, 0))
      return emitOpError("A operand must be a 16x16 f16 wave32 fragment with 8 registers");
    if (!isF16AB(bType, 1))
      return emitOpError("B operand must be a 16x16 f16 wave32 fragment with 8 registers");
    if (!isF32Acc(accType))
      return emitOpError("accumulator must be a 16x16 f32 wave32 fragment with 8 registers");
  }
  if (resultType != accType)
    return emitOpError("result type must match accumulator type");
  return success();
}

LogicalResult FragmentStoreOp::verify() {
  auto fragmentType = cast<FragmentType>(getFragment().getType());
  if (fragmentType.getRole() != 2 ||
      (!fragmentType.getElementType().isInteger(32) &&
       !fragmentType.getElementType().isF32()))
    return emitOpError("only 32-bit accumulator fragment stores are supported for now");
  Type ptrType = getPtr().getType();
  Type ptrElementType;
  if (auto wavePtr = dyn_cast<wave::PtrType>(ptrType)) {
    ptrElementType = wavePtr.getElementType();
  } else if (auto ptrSimdType = dyn_cast<wave::SimdType>(ptrType)) {
    auto wavePtr = dyn_cast<wave::PtrType>(ptrSimdType.getElementType());
    if (!wavePtr)
      return emitOpError("pointer SIMD element type must be a wave pointer");
    if (ptrSimdType.getWidth() != fragmentType.getWaveSize())
      return emitOpError("pointer SIMD width must match fragment wave size");
    ptrElementType = wavePtr.getElementType();
  } else {
    return emitOpError("expected wave pointer operand");
  }
  if (!ptrElementType.isInteger(32) && !ptrElementType.isF32())
    return emitOpError("fragment stores currently require a 32-bit pointer");
  return success();
}

#define GET_OP_CLASSES
#include "mlir/Dialect/Wave/IR/WaveAMDOps.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "mlir/Dialect/Wave/IR/WaveAMDOpsTypes.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "mlir/Dialect/Wave/IR/WaveAMDOpsAttributes.cpp.inc"

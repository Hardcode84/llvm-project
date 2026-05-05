//===- Wave.cpp - Wave dialect ----------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Wave/IR/Wave.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::wave;

#include "mlir/Dialect/Wave/IR/WaveOpsDialect.cpp.inc"

void WaveDialect::initialize() {
  registerAttributes();
  registerTypes();
  addOperations<
#define GET_OP_LIST
#include "mlir/Dialect/Wave/IR/WaveOps.cpp.inc"
      >();
}

void WaveDialect::registerAttributes() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "mlir/Dialect/Wave/IR/WaveOpsAttributes.cpp.inc"
      >();
}

void WaveDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "mlir/Dialect/Wave/IR/WaveOpsTypes.cpp.inc"
      >();
}

LogicalResult WhereOp::verify() {
  auto maskType = dyn_cast<MaskType>(getCondition().getType());
  if (!maskType)
    return emitOpError("condition must be a wave mask");
  if (!getThenRegion().hasOneBlock())
    return emitOpError("then region must have one block");
  if (!getElseRegion().empty() && !getElseRegion().hasOneBlock())
    return emitOpError("otherwise region must have at most one block");
  return success();
}

LogicalResult SplatOp::verify() {
  auto simdType = cast<SimdType>(getResult().getType());
  if (simdType.getElementType() != getSource().getType())
    return emitOpError("source type must match SIMD element type");
  if (simdType.getWidth() != 32 && simdType.getWidth() != 64)
    return emitOpError("only wave32 and wave64 are supported for now");
  return success();
}

LogicalResult BinaryOp::verify() {
  auto lhsType = getLhs().getType();
  auto rhsType = getRhs().getType();
  auto resultType = getResult().getType();
  if (lhsType != rhsType || lhsType != resultType)
    return emitOpError("operands and result must have the same SIMD type");
  return success();
}

LogicalResult CmpIOp::verify() {
  auto lhsType = cast<SimdType>(getLhs().getType());
  auto rhsType = cast<SimdType>(getRhs().getType());
  if (lhsType != rhsType)
    return emitOpError("operands must have the same SIMD type");
  auto resultType = cast<MaskType>(getResult().getType());
  if (lhsType.getWidth() != resultType.getWidth())
    return emitOpError("result mask width must match operand SIMD width");
  return success();
}

LogicalResult BallotOp::verify() {
  unsigned expectedWidth =
      cast<MaskType>(getMask().getType()).getWidth() == 32 ? 32 : 64;
  auto integerType = dyn_cast<IntegerType>(getResult().getType());
  if (!integerType || integerType.getWidth() != expectedWidth)
    return emitOpError("result integer width must match mask width");
  return success();
}

LogicalResult ReadFirstOp::verify() {
  auto simdType = cast<SimdType>(getSource().getType());
  if (simdType.getElementType() != getResult().getType())
    return emitOpError("result type must match SIMD element type");
  return success();
}

LogicalResult StoreOp::verify() {
  auto simdType = cast<SimdType>(getValue().getType());
  Type ptrType = getPtr().getType();
  Type ptrElementType;
  if (auto wavePtr = dyn_cast<PtrType>(ptrType)) {
    ptrElementType = wavePtr.getElementType();
  } else if (auto ptrSimdType = dyn_cast<SimdType>(ptrType)) {
    auto wavePtr = dyn_cast<PtrType>(ptrSimdType.getElementType());
    if (!wavePtr)
      return emitOpError("pointer SIMD element type must be a wave pointer");
    if (ptrSimdType.getWidth() != simdType.getWidth())
      return emitOpError("pointer SIMD width must match value SIMD width");
    ptrElementType = wavePtr.getElementType();
  } else {
    return emitOpError("expected wave pointer operand");
  }

  if (simdType.getElementType() != ptrElementType)
    return emitOpError("SIMD element type must match pointer element type");
  return success();
}

LogicalResult PtrAddOp::verify() {
  Type baseType = getBase().getType();
  Type offsetType = getOffset().getType();
  Type resultType = getResult().getType();

  Type pointerType;
  int64_t pointerWidth = 0;
  if (auto basePtr = dyn_cast<PtrType>(baseType)) {
    pointerType = basePtr;
  } else if (auto baseSimd = dyn_cast<SimdType>(baseType)) {
    if (!isa<PtrType>(baseSimd.getElementType()))
      return emitOpError("base SIMD element type must be a wave pointer");
    pointerType = baseSimd.getElementType();
    pointerWidth = baseSimd.getWidth();
  } else {
    return emitOpError("base must be a wave pointer or SIMD of wave pointers");
  }

  int64_t offsetWidth = 0;
  if (offsetType.isIndex()) {
    offsetWidth = 0;
  } else if (auto intType = dyn_cast<IntegerType>(offsetType)) {
    if (intType.getWidth() != 32 && intType.getWidth() != 64)
      return emitOpError("integer offset must be i32 or i64");
  } else if (auto offsetSimd = dyn_cast<SimdType>(offsetType)) {
    if (!offsetSimd.getElementType().isInteger(32))
      return emitOpError("SIMD offset element type must be i32");
    offsetWidth = offsetSimd.getWidth();
  } else {
    return emitOpError("offset must be index, integer, or i32 SIMD");
  }

  if (pointerWidth && offsetWidth && pointerWidth != offsetWidth)
    return emitOpError("base and offset SIMD widths must match");

  if (offsetWidth || pointerWidth) {
    int64_t width = offsetWidth ? offsetWidth : pointerWidth;
    auto resultSimd = dyn_cast<SimdType>(resultType);
    if (!resultSimd || resultSimd.getElementType() != pointerType ||
        resultSimd.getWidth() != width)
      return emitOpError("result must be a SIMD of the wave pointer type");
  } else if (resultType != pointerType) {
    return emitOpError("result must match base pointer type");
  }
  return success();
}

#define GET_OP_CLASSES
#include "mlir/Dialect/Wave/IR/WaveOps.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "mlir/Dialect/Wave/IR/WaveOpsTypes.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "mlir/Dialect/Wave/IR/WaveOpsAttributes.cpp.inc"

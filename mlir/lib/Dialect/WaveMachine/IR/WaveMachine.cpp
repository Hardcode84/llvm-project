//===- WaveMachine.cpp - WaveMachine dialect --------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/WaveMachine/IR/WaveMachine.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::wavemachine;

#include "mlir/Dialect/WaveMachine/IR/WaveMachineOpsDialect.cpp.inc"
#include "mlir/Dialect/WaveMachine/IR/WaveMachineOpsEnums.cpp.inc"

void WaveMachineDialect::initialize() {
  registerTypes();
  addOperations<
#define GET_OP_LIST
#include "mlir/Dialect/WaveMachine/IR/WaveMachineOps.cpp.inc"
      >();
}

void WaveMachineDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "mlir/Dialect/WaveMachine/IR/WaveMachineOpsTypes.cpp.inc"
      >();
}

#define GET_TYPEDEF_CLASSES
#include "mlir/Dialect/WaveMachine/IR/WaveMachineOpsTypes.cpp.inc"

static bool isRegClassWidth(Type type, RegClass regClass, int64_t width) {
  auto regType = dyn_cast<RegType>(type);
  return regType && regType.getRegClass() == regClass &&
         regType.getWidth() == width;
}

static bool isVGPR(Type type) {
  auto regType = dyn_cast<RegType>(type);
  return regType && regType.getRegClass() == RegClass::VGPR;
}

static LogicalResult verifyVGPRWidth(Operation *op, Value value, int64_t width,
                                     StringRef name) {
  if (!isRegClassWidth(value.getType(), RegClass::VGPR, width))
    return op->emitOpError() << name << " must be !wavemachine.reg<vgpr, "
                             << width << ">";
  return success();
}

LogicalResult VMovB32TupleOp::verify() {
  auto resultType = cast<RegType>(getResult().getType());
  if (auto registers = (*this)->getAttrOfType<IntegerAttr>("registers")) {
    if (registers.getInt() != resultType.getWidth())
      return emitOpError("registers attribute must match result register width");
  }
  return success();
}

LogicalResult WmmaI32_16x16x16_IU8Op::verify() {
  if (failed(verifyVGPRWidth(*this, getOperand(0), 4, "A operand")) ||
      failed(verifyVGPRWidth(*this, getOperand(1), 4, "B operand")) ||
      failed(verifyVGPRWidth(*this, getOperand(2), 8, "accumulator operand")) ||
      failed(verifyVGPRWidth(*this, getResult(), 8, "result")))
    return failure();
  return success();
}

LogicalResult WmmaF32_16x16x16_F16Op::verify() {
  if (failed(verifyVGPRWidth(*this, getOperand(0), 8, "A operand")) ||
      failed(verifyVGPRWidth(*this, getOperand(1), 8, "B operand")) ||
      failed(verifyVGPRWidth(*this, getOperand(2), 8, "accumulator operand")) ||
      failed(verifyVGPRWidth(*this, getResult(), 8, "result")))
    return failure();
  return success();
}

LogicalResult GlobalStoreB32Op::verify() {
  if (getNumResults() > 1)
    return emitOpError("produces at most one memory token");
  return success();
}

LogicalResult BufferStoreB32Op::verify() {
  if (getNumResults() > 1)
    return emitOpError("produces at most one memory token");
  return success();
}

LogicalResult GlobalStoreTupleB32Op::verify() {
  if (getNumResults() > 1)
    return emitOpError("produces at most one memory token");
  auto component = (*this)->getAttrOfType<IntegerAttr>("component");
  if (!component)
    return emitOpError("requires a component attribute");
  auto valueType = cast<RegType>(getOperand(1).getType());
  if (!isVGPR(valueType))
    return emitOpError("value operand must be a VGPR tuple");
  if (component.getInt() < 0 || component.getInt() >= valueType.getWidth())
    return emitOpError("component must select a register in the value tuple");
  return success();
}

#define GET_OP_CLASSES
#include "mlir/Dialect/WaveMachine/IR/WaveMachineOps.cpp.inc"

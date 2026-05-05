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

#define GET_OP_CLASSES
#include "mlir/Dialect/WaveMachine/IR/WaveMachineOps.cpp.inc"

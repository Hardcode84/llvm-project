//===- WaveMachine.h - WaveMachine dialect ----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_WAVEMACHINE_IR_WAVEMACHINE_H
#define MLIR_DIALECT_WAVEMACHINE_IR_WAVEMACHINE_H

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/Dialect/WaveMachine/IR/WaveMachineTraits.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Types.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "mlir/Dialect/WaveMachine/IR/WaveMachineOpsEnums.h.inc"
#include "mlir/Dialect/WaveMachine/IR/WaveMachineOpsDialect.h.inc"

#define GET_TYPEDEF_CLASSES
#include "mlir/Dialect/WaveMachine/IR/WaveMachineOpsTypes.h.inc"

#define GET_OP_CLASSES
#include "mlir/Dialect/WaveMachine/IR/WaveMachineOps.h.inc"

#endif // MLIR_DIALECT_WAVEMACHINE_IR_WAVEMACHINE_H

//===-- WaveMachineOps.td - WaveMachine dialect ------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_WAVEMACHINE_IR_WAVEMACHINEOPS_TD
#define MLIR_DIALECT_WAVEMACHINE_IR_WAVEMACHINEOPS_TD

include "mlir/Dialect/WaveMachine/IR/WaveMachineBase.td"
include "mlir/Dialect/WaveMachine/IR/WaveMachineTypes.td"
include "mlir/IR/OpBase.td"

// The first prototype keeps WaveMachine instructions as unknown operations in a
// registered dialect. This intentionally avoids baking a large target opcode
// inventory into ODS before the machine IR shape settles, while still giving
// every backend stage a textual MLIR boundary.

#endif // MLIR_DIALECT_WAVEMACHINE_IR_WAVEMACHINEOPS_TD

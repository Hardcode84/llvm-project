//===-- WaveMachineBase.td - WaveMachine dialect base -------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_WAVEMACHINE_IR_WAVEMACHINEBASE_TD
#define MLIR_DIALECT_WAVEMACHINE_IR_WAVEMACHINEBASE_TD

include "mlir/IR/DialectBase.td"

def WaveMachine_Dialect : Dialect {
  let name = "wavemachine";
  let cppNamespace = "::mlir::wavemachine";
  let description = [{
    The WaveMachine dialect is an inspectable machine-level IR for the
    experimental Wave-to-AMDGPU backend. It is intentionally lower level than the
    source Wave dialect, but remains MLIR so each backend stage can be printed,
    tested, reduced, and debugged.
  }];

  let useDefaultTypePrinterParser = 1;
  let extraClassDeclaration = [{
    void registerTypes();
  }];
}

#endif // MLIR_DIALECT_WAVEMACHINE_IR_WAVEMACHINEBASE_TD

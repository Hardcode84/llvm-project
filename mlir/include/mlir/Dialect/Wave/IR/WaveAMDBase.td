//===-- WaveAMDBase.td - WaveAMD dialect base -------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_WAVE_IR_WAVEAMDBASE_TD
#define MLIR_DIALECT_WAVE_IR_WAVEAMDBASE_TD

include "mlir/IR/DialectBase.td"

def WaveAMD_Dialect : Dialect {
  let name = "waveamd";
  let cppNamespace = "::mlir::waveamd";
  let description = [{
    The WaveAMD dialect contains AMD-specific extensions to the target-neutral
    Wave dialect.
  }];

  let dependentDialects = [
    "wave::WaveDialect"
  ];
  let useDefaultAttributePrinterParser = 1;
  let useDefaultTypePrinterParser = 1;
  let extraClassDeclaration = [{
    void registerAttributes();
    void registerTypes();
  }];
}

#endif // MLIR_DIALECT_WAVE_IR_WAVEAMDBASE_TD

//===-- WaveBase.td - Wave dialect base -------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_WAVE_IR_WAVEBASE_TD
#define MLIR_DIALECT_WAVE_IR_WAVEBASE_TD

include "mlir/IR/DialectBase.td"

def Wave_Dialect : Dialect {
  let name = "wave";
  let cppNamespace = "::mlir::wave";
  let description = [{
    The Wave dialect is an experimental structured subgroup programming dialect.

    It models a program as uniform scalar control plus lane-varying values and
    explicit lane masks. The dialect is intentionally target-neutral at the IR
    boundary: target-specific lowerings decide whether masks become AMDGPU EXEC,
    NVIDIA predicates/member masks, Intel vISA predicates/execution masks, or
    another backend representation.
  }];

  let dependentDialects = [
    "arith::ArithDialect",
    "gpu::GPUDialect",
    "scf::SCFDialect"
  ];
  let useDefaultAttributePrinterParser = 1;
  let useDefaultTypePrinterParser = 1;
  let extraClassDeclaration = [{
    void registerAttributes();
    void registerTypes();
  }];
}

#endif // MLIR_DIALECT_WAVE_IR_WAVEBASE_TD

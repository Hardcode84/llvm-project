//===-- Passes.td - Wave transformation definitions --------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_WAVE_TRANSFORMS_PASSES_TD
#define MLIR_DIALECT_WAVE_TRANSFORMS_PASSES_TD

include "mlir/Pass/PassBase.td"

def ConvertWaveToGPU : Pass<"convert-wave-to-gpu"> {
  let summary = "Lower Wave dialect operations to GPU and SCF operations";
  let description = [{
    Lowers the experimental Wave dialect to existing structured MLIR dialects.
    `wave.where` becomes `scf.if`, and subgroup operations lower to GPU dialect
    subgroup operations. This keeps examples executable with the existing
    GPU-to-ROCDL pipeline while preserving a separate source-level Wave dialect.
  }];

  let dependentDialects = [
    "arith::ArithDialect",
    "gpu::GPUDialect",
    "scf::SCFDialect"
  ];
}

def ConvertWaveToROCDL : Pass<"convert-wave-to-rocdl"> {
  let summary = "Lower Wave dialect operations to ROCDL and SCF operations";
  let description = [{
    Lowers Wave dialect operations directly to ROCDL intrinsics and SCF
    structured control. This is the AMDGPU-oriented path for standalone Wave
    dialect examples that do not use `gpu.module`.
  }];

  let dependentDialects = [
    "arith::ArithDialect",
    "ROCDL::ROCDLDialect",
    "scf::SCFDialect"
  ];
}

#endif // MLIR_DIALECT_WAVE_TRANSFORMS_PASSES_TD

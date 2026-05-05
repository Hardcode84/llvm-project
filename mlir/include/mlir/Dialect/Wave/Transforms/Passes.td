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

def ConvertWaveAMDToWaveMachine : Pass<"waveamd-to-wavemachine", "ModuleOp"> {
  let summary = "Select Wave/WaveAMD operations into inspectable WaveMachine IR";
  let description = [{
    Lowers source-level Wave operations to generic WaveMachine dialect
    operations. This is the explicit backend boundary: the result is still MLIR,
    but the values are selected into SGPR/VGPR-like register classes and
    machine-like operations.
  }];

  let dependentDialects = [
    "func::FuncDialect",
    "waveamd::WaveAMDDialect",
    "wavemachine::WaveMachineDialect"
  ];
}

def WaveAMDABILowering
    : Pass<"waveamd-abi-lowering", "ModuleOp"> {
  let summary = "Lower WaveMachine function arguments to the AMDGPU kernel ABI";
  let dependentDialects = [
    "wavemachine::WaveMachineDialect"
  ];
}

def WaveAMDHazardWaits
    : Pass<"waveamd-insert-hazard-waits", "ModuleOp"> {
  let summary = "Insert WaveMachine waitcnt and hazard instructions";
  let dependentDialects = [
    "wavemachine::WaveMachineDialect"
  ];
}

def WaveAMDTicketWaits
    : Pass<"waveamd-insert-ticket-waits", "ModuleOp"> {
  let summary = "Insert WaveMachine waitcnt instructions using ticket scoreboards";
  let description = [{
    Tracks async memory operations with per-counter tickets and inserts the
    minimum waitcnt threshold required by SSA uses. This is the waitcnt pass
    intended for the staged WaveMachine backend; pipeline hazards remain a
    separate concern.
  }];
  let dependentDialects = [
    "wavemachine::WaveMachineDialect"
  ];
}

def WaveAMDRegAlloc
    : Pass<"waveamd-reg-alloc", "ModuleOp"> {
  let summary = "Allocate WaveMachine virtual registers to physical registers";
  let dependentDialects = [
    "wavemachine::WaveMachineDialect"
  ];
}

def WaveAMDResourceInfo
    : Pass<"waveamd-resource-info", "ModuleOp"> {
  let summary = "Annotate WaveMachine functions with resource usage";
  let dependentDialects = [
    "wavemachine::WaveMachineDialect"
  ];
}

def WaveAMDMetadata
    : Pass<"waveamd-metadata", "ModuleOp"> {
  let summary = "Attach AMDGPU metadata information to WaveMachine IR";
  let dependentDialects = [
    "wavemachine::WaveMachineDialect"
  ];
}

#endif // MLIR_DIALECT_WAVE_TRANSFORMS_PASSES_TD

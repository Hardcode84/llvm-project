//===- AMDGPU.h - Wave to AMDGPU backend ------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TARGET_WAVE_AMDGPU_H
#define MLIR_TARGET_WAVE_AMDGPU_H

#include "mlir/Support/LLVM.h"

namespace mlir {
class Operation;

namespace wave {

/// Emit AMDGPU assembly for the Wave dialect MVP.
///
/// This backend consumes Wave operations directly. It is intentionally small and
/// currently targets straight-line wave functions plus structured `wave.where`
/// regions over 32-lane masks.
LogicalResult translateWaveToAMDGPU(Operation *op, raw_ostream &os);

/// Register the `wave-to-amdgpu-asm` mlir-translate entry point.
void registerWaveToAMDGPUTranslation();

} // namespace wave
} // namespace mlir

#endif // MLIR_TARGET_WAVE_AMDGPU_H

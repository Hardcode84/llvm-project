//===- AMDGPUMFMAIGroupLP.h - AMDGPU MFMA IGroupLP --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_AMDGPU_AMDGPUMFMAIGROUPLP_H
#define LLVM_LIB_TARGET_AMDGPU_AMDGPUMFMAIGROUPLP_H

#include "llvm/CodeGen/ScheduleDAGMutation.h"
#include <memory>
#include "llvm/Support/Compiler.h"

LLVM_NAMESPACE_BEGIN

namespace AMDGPU {
// The current phase of instruction scheduling
enum class SchedulingPhase { Initial, PreRAReentry, PostRA };

/// Operand 0 immediate for IGLP_OPT pseudo instructions.
enum IGLPStrategyID : int {
  MFMASmallGemmOptID = 0,
  MFMASmallGemmSingleWaveOptID = 1,
  MFMAExpInterleaveID = 2,
  MFMAExpSimpleInterleaveID = 3,
};
} // namespace AMDGPU

std::unique_ptr<ScheduleDAGMutation>
createIGroupLPDAGMutation(AMDGPU::SchedulingPhase Phase);

LLVM_NAMESPACE_END // namespace llvm

#endif // LLVM_LIB_TARGET_AMDGPU_AMDGPUMFMAIGROUPLP_H

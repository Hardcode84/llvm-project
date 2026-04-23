//===- ARMMacroFusion.h - ARM Macro Fusion ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file This file contains the ARM definition of the DAG scheduling mutation
/// to pair instructions back to back.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_ARM_ARMMACROFUSION_H
#define LLVM_LIB_TARGET_ARM_ARMMACROFUSION_H

#include "llvm/CodeGen/MachineScheduler.h"
#include "llvm/Support/Compiler.h"

LLVM_NAMESPACE_BEGIN

/// Note that you have to add:
///   DAG.addMutation(createARMMacroFusionDAGMutation());
/// to ARMTargetMachine::createMachineScheduler() to have an effect.
std::unique_ptr<ScheduleDAGMutation> createARMMacroFusionDAGMutation();

LLVM_NAMESPACE_END // llvm

#endif

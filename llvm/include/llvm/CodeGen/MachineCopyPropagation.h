//===- llvm/CodeGen/MachineCopyPropagation.h --------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_MACHINECOPYPROPAGATION_H
#define LLVM_CODEGEN_MACHINECOPYPROPAGATION_H

#include "llvm/CodeGen/MachinePassManager.h"
#include "llvm/Support/Compiler.h"

LLVM_NAMESPACE_BEGIN

class MachineCopyPropagationPass
    : public PassInfoMixin<MachineCopyPropagationPass> {
  bool UseCopyInstr;

public:
  MachineCopyPropagationPass(bool UseCopyInstr = false)
      : UseCopyInstr(UseCopyInstr) {}

  PreservedAnalyses run(MachineFunction &MF,
                        MachineFunctionAnalysisManager &MFAM);

  MachineFunctionProperties getRequiredProperties() const {
    return MachineFunctionProperties().setNoVRegs();
  }
};

LLVM_NAMESPACE_END // namespace llvm

#endif // LLVM_CODEGEN_MACHINECOPYPROPAGATION_H

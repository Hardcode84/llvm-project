//===- llvm/CodeGen/TwoAddressInstructionPass.h -----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_TWOADDRESSINSTRUCTIONPASS_H
#define LLVM_CODEGEN_TWOADDRESSINSTRUCTIONPASS_H

#include "llvm/CodeGen/MachinePassManager.h"
#include "llvm/Support/Compiler.h"

LLVM_NAMESPACE_BEGIN

class TwoAddressInstructionPass
    : public PassInfoMixin<TwoAddressInstructionPass> {
public:
  PreservedAnalyses run(MachineFunction &MF,
                        MachineFunctionAnalysisManager &MFAM);
  MachineFunctionProperties getSetProperties() const {
    return MachineFunctionProperties().setTiedOpsRewritten();
  }
};

LLVM_NAMESPACE_END // namespace llvm

#endif // LLVM_CODEGEN_TWOADDRESSINSTRUCTIONPASS_H

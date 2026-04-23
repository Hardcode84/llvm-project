//===- llvm/CodeGen/RegisterCoalescerPass.h ---------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_REGISTER_COALESCERPASS_H
#define LLVM_CODEGEN_REGISTER_COALESCERPASS_H

#include "llvm/CodeGen/MachinePassManager.h"
#include "llvm/Support/Compiler.h"

LLVM_NAMESPACE_BEGIN
class RegisterCoalescerPass : public PassInfoMixin<RegisterCoalescerPass> {
public:
  PreservedAnalyses run(MachineFunction &MF,
                        MachineFunctionAnalysisManager &MFAM);

  MachineFunctionProperties getClearedProperties() const {
    return MachineFunctionProperties().setIsSSA();
  }
};

LLVM_NAMESPACE_END // namespace llvm

#endif // LLVM_CODEGEN_REGISTER_COALESCERPASS_H

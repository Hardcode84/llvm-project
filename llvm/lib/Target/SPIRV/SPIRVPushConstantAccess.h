//===- SPIRVPushConstantAccess.h - Translate Push constant loads ----------*-
// C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

#ifndef LLVM_LIB_TARGET_SPIRV_SPIRVPUSHCONSTANTACCESS_H
#define LLVM_LIB_TARGET_SPIRV_SPIRVPUSHCONSTANTACCESS_H

#include "SPIRVTargetMachine.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Support/Compiler.h"

LLVM_NAMESPACE_BEGIN

class SPIRVPushConstantAccess : public PassInfoMixin<SPIRVPushConstantAccess> {
  const SPIRVTargetMachine &TM;

public:
  SPIRVPushConstantAccess(const SPIRVTargetMachine &TM) : TM(TM) {}
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM);
};

LLVM_NAMESPACE_END // namespace llvm

#endif // LLVM_LIB_TARGET_SPIRV_SPIRVPUSHCONSTANTACCESS_H

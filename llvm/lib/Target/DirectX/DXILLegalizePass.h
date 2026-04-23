//===- DXILLegalizePass.h - Legalizes llvm IR for DXIL --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===---------------------------------------------------------------------===//

#ifndef LLVM_TARGET_DIRECTX_LEGALIZE_H
#define LLVM_TARGET_DIRECTX_LEGALIZE_H

#include "llvm/IR/PassManager.h"
#include "llvm/Support/Compiler.h"

LLVM_NAMESPACE_BEGIN

class DXILLegalizePass : public PassInfoMixin<DXILLegalizePass> {
public:
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &FAM);
};
LLVM_NAMESPACE_END // namespace llvm

#endif // LLVM_TARGET_DIRECTX_LEGALIZE_H

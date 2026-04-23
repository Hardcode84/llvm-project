//===-- InlineAsmPrepare - Prepare inline asm for code gen ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_INLINEASMPREPARE_H
#define LLVM_CODEGEN_INLINEASMPREPARE_H

#include "llvm/IR/PassManager.h"
#include "llvm/Support/Compiler.h"

LLVM_NAMESPACE_BEGIN

class InlineAsmPreparePass : public PassInfoMixin<InlineAsmPreparePass> {
public:
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &FAM);
};

LLVM_NAMESPACE_END // namespace llvm

#endif // LLVM_CODEGEN_INLINEASMPREPARE_H

//===- LoopTermFold.h - Loop Term Fold Pass ---------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_SCALAR_LOOPTERMFOLD_H
#define LLVM_TRANSFORMS_SCALAR_LOOPTERMFOLD_H

#include "llvm/Analysis/LoopAnalysisManager.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Support/Compiler.h"

LLVM_NAMESPACE_BEGIN

class Loop;
class LPMUpdater;

class LoopTermFoldPass : public PassInfoMixin<LoopTermFoldPass> {
public:
  PreservedAnalyses run(Loop &L, LoopAnalysisManager &AM,
                        LoopStandardAnalysisResults &AR, LPMUpdater &U);
};

LLVM_NAMESPACE_END // end namespace llvm

#endif // LLVM_TRANSFORMS_SCALAR_LOOPTERMFOLD_H

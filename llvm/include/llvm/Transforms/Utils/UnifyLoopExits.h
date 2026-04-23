//===- UnifyLoopExits.h - Redirect exiting edges to one block -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_UTILS_UNIFYLOOPEXITS_H
#define LLVM_TRANSFORMS_UTILS_UNIFYLOOPEXITS_H

#include "llvm/IR/PassManager.h"
#include "llvm/Support/Compiler.h"

LLVM_NAMESPACE_BEGIN

class UnifyLoopExitsPass : public PassInfoMixin<UnifyLoopExitsPass> {
public:
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM);
};
LLVM_NAMESPACE_END // namespace llvm

#endif // LLVM_TRANSFORMS_UTILS_UNIFYLOOPEXITS_H

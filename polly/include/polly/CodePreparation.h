//===- polly/ScopPreparation.h - Code preparation pass ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Prepare the Function for polyhedral codegeneration.
//
//===----------------------------------------------------------------------===//

#ifndef POLLY_CODEPREPARATION_H
#define POLLY_CODEPREPARATION_H

#include "llvm/Support/Compiler.h"
LLVM_NAMESPACE_BEGIN
class DominatorTree;
class Function;
class LoopInfo;
class RegionInfo;
LLVM_NAMESPACE_END // namespace llvm

namespace polly {
bool runCodePreparation(llvm::Function &F, llvm::DominatorTree *DT,
                        llvm::LoopInfo *LI, llvm::RegionInfo *RI);
} // namespace polly

#endif /* POLLY_CODEPREPARATION_H */

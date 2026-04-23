//===- TransactionSave.cpp - Save the IR state ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Vectorize/SandboxVectorizer/Passes/TransactionSave.h"
#include "llvm/Support/InstructionCost.h"
#include "llvm/Transforms/Vectorize/SandboxVectorizer/Debug.h"
#include "llvm/Support/Compiler.h"

LLVM_NAMESPACE_BEGIN
namespace sandboxir {

bool TransactionSave::runOnRegion(Region &Rgn, const Analyses &A) {
  LLVM_DEBUG(dbgs() << DEBUG_PREFIX << "*** Save Transaction ***\n");
  Rgn.getContext().save();
  return false;
}

}
LLVM_NAMESPACE_END // namespace llvm::sandboxir

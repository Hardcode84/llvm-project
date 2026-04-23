//===- DXILMemIntrinsics.h -  Eliminate Memory Intrinsics -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TARGET_DIRECTX_DXILMEMINTRINSICS_H
#define LLVM_TARGET_DIRECTX_DXILMEMINTRINSICS_H

#include "llvm/IR/PassManager.h"
#include "llvm/Pass.h"
#include "llvm/Support/Compiler.h"

LLVM_NAMESPACE_BEGIN

/// Transform all llvm memory intrinsics to explicit loads and stores.
class DXILMemIntrinsics : public PassInfoMixin<DXILMemIntrinsics> {
public:
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &);
};

LLVM_NAMESPACE_END // namespace llvm

#endif // LLVM_TARGET_DIRECTX_DXILMEMINTRINSICS_H

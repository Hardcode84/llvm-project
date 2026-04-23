//===- DXILResourceAccess.h - Resource access via load/store ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// \file Pass for replacing pointers to DXIL resources with load and store
// operations.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_DIRECTX_DXILRESOURCEACCESS_H
#define LLVM_LIB_TARGET_DIRECTX_DXILRESOURCEACCESS_H

#include "llvm/IR/PassManager.h"
#include "llvm/Support/Compiler.h"

LLVM_NAMESPACE_BEGIN

class DXILResourceAccess : public PassInfoMixin<DXILResourceAccess> {
public:
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM);
};

LLVM_NAMESPACE_END // namespace llvm

#endif // LLVM_LIB_TARGET_DIRECTX_DXILRESOURCEACCESS_H

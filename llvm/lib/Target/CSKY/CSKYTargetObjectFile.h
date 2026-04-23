//===-- CSKYTargetObjectFile.h - CSKY Object Info -*- C++ ---------------*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_CSKY_CSKYTARGETOBJECTFILE_H
#define LLVM_LIB_TARGET_CSKY_CSKYTARGETOBJECTFILE_H

#include "llvm/CodeGen/TargetLoweringObjectFileImpl.h"
#include "llvm/Target/TargetLoweringObjectFile.h"
#include "llvm/Support/Compiler.h"

LLVM_NAMESPACE_BEGIN

class CSKYELFTargetObjectFile : public TargetLoweringObjectFileELF {
public:
  void Initialize(MCContext &Ctx, const TargetMachine &TM) override;
};

LLVM_NAMESPACE_END // namespace llvm

#endif // LLVM_LIB_TARGET_CSKY_CSKYTARGETOBJECTFILE_H

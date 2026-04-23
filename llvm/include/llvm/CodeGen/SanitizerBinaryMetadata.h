//===- llvm/CodeGen/SanitizerBinaryMetadata.h -------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_SANITIZERBINARYMETADATA_H
#define LLVM_CODEGEN_SANITIZERBINARYMETADATA_H

#include "llvm/CodeGen/MachinePassManager.h"
#include "llvm/Support/Compiler.h"

LLVM_NAMESPACE_BEGIN

class MachineSanitizerBinaryMetadataPass
    : public PassInfoMixin<MachineSanitizerBinaryMetadataPass> {
public:
  PreservedAnalyses run(MachineFunction &MF,
                        MachineFunctionAnalysisManager &MFAM);
  static bool isRequired() { return true; }
};

LLVM_NAMESPACE_END // namespace llvm

#endif // LLVM_CODEGEN_SANITIZERBINARYMETADATA_H

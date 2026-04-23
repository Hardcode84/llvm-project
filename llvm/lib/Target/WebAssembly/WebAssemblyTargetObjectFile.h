//===-- WebAssemblyTargetObjectFile.h - WebAssembly Object Info -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file declares the WebAssembly-specific subclass of
/// TargetLoweringObjectFile.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_WEBASSEMBLY_WEBASSEMBLYTARGETOBJECTFILE_H
#define LLVM_LIB_TARGET_WEBASSEMBLY_WEBASSEMBLYTARGETOBJECTFILE_H

#include "llvm/CodeGen/TargetLoweringObjectFileImpl.h"
#include "llvm/Support/Compiler.h"

LLVM_NAMESPACE_BEGIN

class WebAssemblyTargetObjectFile final : public TargetLoweringObjectFileWasm {
public:
  void Initialize(MCContext &Ctx, const TargetMachine &TM) override;
};

LLVM_NAMESPACE_END // end namespace llvm

#endif

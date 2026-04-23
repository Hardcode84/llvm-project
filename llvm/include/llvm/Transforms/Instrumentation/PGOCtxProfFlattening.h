//===-- PGOCtxProfFlattening.h - Contextual Instr. Flattening ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the PGOCtxProfFlattening class.
//
//===----------------------------------------------------------------------===//
#ifndef LLVM_TRANSFORMS_INSTRUMENTATION_PGOCTXPROFFLATTENING_H
#define LLVM_TRANSFORMS_INSTRUMENTATION_PGOCTXPROFFLATTENING_H

#include "llvm/IR/PassManager.h"
#include "llvm/Support/Compiler.h"
LLVM_NAMESPACE_BEGIN

class PGOCtxProfFlatteningPass
    : public PassInfoMixin<PGOCtxProfFlatteningPass> {
  const bool IsPreThinlink;

public:
  explicit PGOCtxProfFlatteningPass(bool IsPreThinlink)
      : IsPreThinlink(IsPreThinlink) {}
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &MAM);
};
LLVM_NAMESPACE_END // namespace llvm
#endif

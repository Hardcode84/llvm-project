//===- bolt/Passes/Hugify.h -------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef BOLT_PASSES_HUGIFY_H
#define BOLT_PASSES_HUGIFY_H

#include "bolt/Passes/BinaryPasses.h"
#include "llvm/Support/Compiler.h"

LLVM_NAMESPACE_BEGIN
namespace bolt {

class HugePage : public BinaryFunctionPass {
public:
  HugePage(const cl::opt<bool> &PrintPass) : BinaryFunctionPass(PrintPass) {}

  Error runOnFunctions(BinaryContext &BC) override;

  const char *getName() const override { return "HugePage"; }
};

} // namespace bolt
LLVM_NAMESPACE_END // namespace llvm

#endif

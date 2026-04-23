//===- llvm/CodeGen/GlobalISel/InstructionSelector.cpp --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/GlobalISel/InstructionSelector.h"
#include "llvm/Support/Compiler.h"

LLVM_NAMESPACE_BEGIN

// vtable anchor
InstructionSelector::~InstructionSelector() = default;

LLVM_NAMESPACE_END // namespace llvm

//===- bolt/Passes/LivenessAnalysis.cpp -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bolt/Passes/LivenessAnalysis.h"
#include "llvm/Support/Compiler.h"

LLVM_NAMESPACE_BEGIN
namespace bolt {

LivenessAnalysis::~LivenessAnalysis() {}

} // end namespace bolt
LLVM_NAMESPACE_END // end namespace llvm

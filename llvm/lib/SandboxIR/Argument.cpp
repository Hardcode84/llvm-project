//===- Argument.cpp - The function Argument class of Sandbox IR -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/SandboxIR/Argument.h"
#include "llvm/Support/Compiler.h"

LLVM_NAMESPACE_BEGIN
namespace sandboxir {

#ifndef NDEBUG
void Argument::printAsOperand(raw_ostream &OS) const {
  printAsOperandCommon(OS);
}
void Argument::dumpOS(raw_ostream &OS) const {
  dumpCommonPrefix(OS);
  dumpCommonSuffix(OS);
}
#endif // NDEBUG

}
LLVM_NAMESPACE_END // namespace llvm::sandboxir

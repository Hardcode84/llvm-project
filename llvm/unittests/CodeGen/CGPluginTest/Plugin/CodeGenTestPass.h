//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CODEGEN_TEST_PASS
#define CODEGEN_TEST_PASS

#include <llvm/CodeGen/MachineFunctionPass.h>
#include "llvm/Support/Compiler.h"

using namespace llvm;

LLVM_NAMESPACE_BEGIN
void initializeCodeGenTestPass(PassRegistry &);
LLVM_NAMESPACE_END // namespace llvm

class CodeGenTest : public MachineFunctionPass {
public:
  static char ID;

  CodeGenTest();

  bool runOnMachineFunction(MachineFunction &MF) override;

  StringRef getPassName() const override;

  static std::function<void()> RunCallback;
};

#endif // CODEGEN_TEST_PASS

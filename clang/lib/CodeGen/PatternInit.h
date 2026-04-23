//===- PatternInit - Pattern initialization ---------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_LIB_CODEGEN_PATTERNINIT_H
#define LLVM_CLANG_LIB_CODEGEN_PATTERNINIT_H


#include "llvm/Support/Compiler.h"
LLVM_NAMESPACE_BEGIN
class Constant;
class Type;
LLVM_NAMESPACE_END // namespace llvm

namespace clang {
namespace CodeGen {

class CodeGenModule;

llvm::Constant *initializationPatternFor(CodeGenModule &, llvm::Type *);

} // end namespace CodeGen
} // end namespace clang

#endif

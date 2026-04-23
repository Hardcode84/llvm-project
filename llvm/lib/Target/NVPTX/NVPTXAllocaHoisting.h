//===-- AllocaHoisting.h - Hosist allocas to the entry block ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Hoist the alloca instructions in the non-entry blocks to the entry blocks.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_NVPTX_NVPTXALLOCAHOISTING_H
#define LLVM_LIB_TARGET_NVPTX_NVPTXALLOCAHOISTING_H


#include "llvm/Support/Compiler.h"
LLVM_NAMESPACE_BEGIN
class FunctionPass;

FunctionPass *createAllocaHoisting();
LLVM_NAMESPACE_END // end namespace llvm

#endif

//===- llvm/Support/Unix/COM.inc - Unix COM Implementation -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the Unix portion of COM support.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
//=== WARNING: Implementation here must contain only generic UNIX code that
//===          is guaranteed to work on *all* UNIX variants.
//===----------------------------------------------------------------------===//

#include "llvm/Support/Compiler.h"
LLVM_NAMESPACE_BEGIN
namespace sys {

InitializeCOMRAII::InitializeCOMRAII(COMThreadingMode Threading,
                                     bool SpeedOverMemory) {}

InitializeCOMRAII::~InitializeCOMRAII() = default;
} // namespace sys
LLVM_NAMESPACE_END // namespace llvm

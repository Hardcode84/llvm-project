//===-- DebugerSupport.h - Utils for enabling debugger support --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Utilities for enabling debugger support.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_EXECUTIONENGINE_ORC_DEBUGGERSUPPORT_H
#define LLVM_EXECUTIONENGINE_ORC_DEBUGGERSUPPORT_H

#include "llvm/Support/Compiler.h"
#include "llvm/Support/Error.h"

LLVM_NAMESPACE_BEGIN
namespace orc {

class LLJIT;

LLVM_ABI Error enableDebuggerSupport(LLJIT &J);

} // namespace orc
LLVM_NAMESPACE_END // namespace llvm

#endif // LLVM_EXECUTIONENGINE_ORC_DEBUGGERSUPPORT_H

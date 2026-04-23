//===-- M68kTargetInfo.h - M68k Target Implementation -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_M68K_TARGETINFO_M68KTARGETINFO_H
#define LLVM_LIB_TARGET_M68K_TARGETINFO_M68KTARGETINFO_H


#include "llvm/Support/Compiler.h"
LLVM_NAMESPACE_BEGIN
class Target;

Target &getTheM68kTarget();
LLVM_NAMESPACE_END // namespace llvm

#endif // LLVM_LIB_TARGET_M68K_TARGETINFO_M68KTARGETINFO_H

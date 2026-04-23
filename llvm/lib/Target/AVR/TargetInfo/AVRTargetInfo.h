//===-- AVRTargetInfo.h - AVR Target Implementation -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_AVR_TARGET_INFO_H
#define LLVM_AVR_TARGET_INFO_H


#include "llvm/Support/Compiler.h"
LLVM_NAMESPACE_BEGIN
class Target;

Target &getTheAVRTarget();
LLVM_NAMESPACE_END // namespace llvm

#endif // LLVM_AVR_TARGET_INFO_H

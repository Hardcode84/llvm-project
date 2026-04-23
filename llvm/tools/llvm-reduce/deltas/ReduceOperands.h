//===------------------------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TOOLS_LLVM_REDUCE_DELTAS_REDUCEOPERANDS_H
#define LLVM_TOOLS_LLVM_REDUCE_DELTAS_REDUCEOPERANDS_H

#include "Delta.h"
#include "llvm/Support/Compiler.h"

LLVM_NAMESPACE_BEGIN
void reduceOperandsOneDeltaPass(Oracle &, ReducerWorkItem &);
void reduceOperandsZeroDeltaPass(Oracle &, ReducerWorkItem &);
void reduceOperandsNaNDeltaPass(Oracle &, ReducerWorkItem &);
void reduceOperandsPoisonDeltaPass(Oracle &, ReducerWorkItem &);
LLVM_NAMESPACE_END // namespace llvm

#endif

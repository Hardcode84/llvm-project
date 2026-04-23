//===- OptEmitter.h - Helper for emitting options ---------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_UTILS_TABLEGEN_COMMON_OPTEMITTER_H
#define LLVM_UTILS_TABLEGEN_COMMON_OPTEMITTER_H


#include "llvm/Support/Compiler.h"
LLVM_NAMESPACE_BEGIN
class Record;
/// Return true of Option record \p A is ordered before \p B.
bool IsOptionRecordsLess(const Record *A, const Record *B);
LLVM_NAMESPACE_END // namespace llvm

#endif // LLVM_UTILS_TABLEGEN_COMMON_OPTEMITTER_H

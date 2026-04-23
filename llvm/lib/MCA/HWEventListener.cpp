//===----------------------- HWEventListener.cpp ----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file
///
/// This file defines a vtable anchor for class HWEventListener.
///
//===----------------------------------------------------------------------===//

#include "llvm/MCA/HWEventListener.h"
#include "llvm/Support/Compiler.h"

LLVM_NAMESPACE_BEGIN
namespace mca {

// Anchor the vtable here.
void HWEventListener::anchor() {}
} // namespace mca
LLVM_NAMESPACE_END // namespace llvm

//===----------------------- View.cpp ---------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file
///
/// This file defines the virtual anchor method in View.h to pin the vtable.
///
//===----------------------------------------------------------------------===//

#include "llvm/MCA/View.h"
#include "llvm/Support/Compiler.h"

LLVM_NAMESPACE_BEGIN
namespace mca {

void View::anchor() {}

} // namespace mca
LLVM_NAMESPACE_END // namespace llvm

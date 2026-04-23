//===------------------------- HardwareUnit.cpp -----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file
///
/// This file defines the anchor for the base class that describes
/// simulated hardware units.
///
//===----------------------------------------------------------------------===//

#include "llvm/MCA/HardwareUnits/HardwareUnit.h"
#include "llvm/Support/Compiler.h"

LLVM_NAMESPACE_BEGIN
namespace mca {

// Pin the vtable with this method.
HardwareUnit::~HardwareUnit() = default;

} // namespace mca
LLVM_NAMESPACE_END // namespace llvm

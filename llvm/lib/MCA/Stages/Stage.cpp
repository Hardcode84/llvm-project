//===---------------------- Stage.cpp ---------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file
///
/// This file defines a stage.
/// A chain of stages compose an instruction pipeline.
///
//===----------------------------------------------------------------------===//

#include "llvm/MCA/Stages/Stage.h"
#include "llvm/Support/Compiler.h"

LLVM_NAMESPACE_BEGIN
namespace mca {

// Pin the vtable here in the implementation file.
Stage::~Stage() = default;

void Stage::addListener(HWEventListener *Listener) {
  Listeners.insert(Listener);
}

char InstStreamPause::ID = 0;
} // namespace mca
LLVM_NAMESPACE_END // namespace llvm

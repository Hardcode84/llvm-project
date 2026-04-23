//===- opt.cpp - The LLVM Modular Optimizer -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Optimizations may be specified an arbitrary number of times on the command
// line, They are run in the order specified.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/ArrayRef.h"
#include <functional>
#include "llvm/Support/Compiler.h"

using namespace llvm;

LLVM_NAMESPACE_BEGIN
class PassBuilder;
LLVM_NAMESPACE_END

extern "C" int
optMain(int argc, char **argv,
        ArrayRef<std::function<void(PassBuilder &)>> PassBuilderCallbacks);

int main(int argc, char **argv) { return optMain(argc, argv, {}); }

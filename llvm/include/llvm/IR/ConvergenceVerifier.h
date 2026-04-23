//===- ConvergenceVerifier.h - Verify convergenctrl -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file
///
/// This file declares the LLVM IR specialization of the
/// GenericConvergenceVerifier template.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_IR_CONVERGENCEVERIFIER_H
#define LLVM_IR_CONVERGENCEVERIFIER_H

#include "llvm/ADT/GenericConvergenceVerifier.h"
#include "llvm/IR/SSAContext.h"
#include "llvm/Support/Compiler.h"

LLVM_NAMESPACE_BEGIN

using ConvergenceVerifier = GenericConvergenceVerifier<SSAContext>;

LLVM_NAMESPACE_END // namespace llvm

#endif // LLVM_IR_CONVERGENCEVERIFIER_H

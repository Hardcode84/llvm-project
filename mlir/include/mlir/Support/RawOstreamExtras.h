//===- RawOstreamExtras.h - Extensions to LLVM's raw_ostream.h --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/Compiler.h"
#include "mlir/Support/ABINamespace.h"
LLVM_NAMESPACE_BEGIN
class raw_ostream;
LLVM_NAMESPACE_END // namespace llvm

MLIR_NAMESPACE_BEGIN
/// Returns a raw output stream that simply discards the output, but in a
/// thread-safe manner. Similar to llvm::nulls.
llvm::raw_ostream &thread_safe_nulls();
MLIR_NAMESPACE_END // namespace mlir

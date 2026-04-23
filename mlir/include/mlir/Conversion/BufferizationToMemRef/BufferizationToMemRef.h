//===- BufferizationToMemRef.h - Bufferization to MemRef conversion -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_CONVERSION_BUFFERIZATIONTOMEMREF_BUFFERIZATIONTOMEMREF_H
#define MLIR_CONVERSION_BUFFERIZATIONTOMEMREF_BUFFERIZATIONTOMEMREF_H

#include "mlir/Pass/Pass.h"
#include <memory>
#include "mlir/Support/ABINamespace.h"

MLIR_NAMESPACE_BEGIN
class ModuleOp;

#define GEN_PASS_DECL_CONVERTBUFFERIZATIONTOMEMREFPASS
#include "mlir/Conversion/Passes.h.inc"

MLIR_NAMESPACE_END // namespace mlir

#endif // MLIR_CONVERSION_BUFFERIZATIONTOMEMREF_BUFFERIZATIONTOMEMREF_H

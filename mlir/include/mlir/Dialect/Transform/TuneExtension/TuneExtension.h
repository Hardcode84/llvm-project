//===- TuneExtension.h - Tune extension for Transform dialect ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_TRANSFORM_TUNEEXTENSION_TUNEEXTENSION_H
#define MLIR_DIALECT_TRANSFORM_TUNEEXTENSION_TUNEEXTENSION_H


#include "mlir/Support/ABINamespace.h"
MLIR_NAMESPACE_BEGIN
class DialectRegistry;

namespace transform {
/// Registers the tune extension of the Transform dialect in the given registry.
void registerTuneExtension(DialectRegistry &dialectRegistry);
} // namespace transform
MLIR_NAMESPACE_END // namespace mlir

#endif // MLIR_DIALECT_TRANSFORM_TUNEEXTENSION_TUNEEXTENSION_H

//===- ReconcileUnrealizedCasts.h - Pass entrypoint -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_CONVERSION_RECONCILEUNREALIZEDCASTS_RECONCILEUNREALIZEDCASTS_H_
#define MLIR_CONVERSION_RECONCILEUNREALIZEDCASTS_RECONCILEUNREALIZEDCASTS_H_

#include <memory>
#include "mlir/Support/ABINamespace.h"

MLIR_NAMESPACE_BEGIN
class Pass;
class RewritePatternSet;

#define GEN_PASS_DECL_RECONCILEUNREALIZEDCASTSPASS
#include "mlir/Conversion/Passes.h.inc"

MLIR_NAMESPACE_END // namespace mlir

#endif // MLIR_CONVERSION_RECONCILEUNREALIZEDCASTS_RECONCILEUNREALIZEDCASTS_H_

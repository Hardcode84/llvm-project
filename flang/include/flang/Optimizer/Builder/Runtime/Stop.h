//===-- Stop.h - generate stop runtime API calls ----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_OPTIMIZER_BUILDER_RUNTIME_STOP_H
#define FORTRAN_OPTIMIZER_BUILDER_RUNTIME_STOP_H


#include "llvm/Support/Compiler.h"
#include "mlir/Support/ABINamespace.h"
LLVM_NAMESPACE_BEGIN
class StringRef;
LLVM_NAMESPACE_END

MLIR_NAMESPACE_BEGIN
class Value;
class Location;
MLIR_NAMESPACE_END // namespace mlir

namespace fir {
class FirOpBuilder;
}

namespace fir::runtime {

/// Generate call to EXIT intrinsic runtime routine.
void genExit(fir::FirOpBuilder &, mlir::Location, mlir::Value status);

/// Generate call to ABORT intrinsic runtime routine.
void genAbort(fir::FirOpBuilder &, mlir::Location);

/// Generate call to crash the program with an error message when detecting
/// an invalid situation at runtime.
void genReportFatalUserError(fir::FirOpBuilder &, mlir::Location,
                             llvm::StringRef message);

} // namespace fir::runtime
#endif // FORTRAN_OPTIMIZER_BUILDER_RUNTIME_STOP_H

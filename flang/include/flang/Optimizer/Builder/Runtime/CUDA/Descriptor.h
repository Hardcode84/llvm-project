//===-- Descriptor.h - CUDA descritpor runtime API calls --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_OPTIMIZER_BUILDER_RUNTIME_CUDA_DESCRIPTOR_H_
#define FORTRAN_OPTIMIZER_BUILDER_RUNTIME_CUDA_DESCRIPTOR_H_

#include "mlir/IR/Value.h"
#include "mlir/Support/ABINamespace.h"

MLIR_NAMESPACE_BEGIN
class Location;
MLIR_NAMESPACE_END // namespace mlir

namespace fir {
class FirOpBuilder;
}

namespace fir::runtime::cuda {

/// Generate runtime call to sync the doublce descriptor referenced by
/// \p hostPtr.
void genSyncGlobalDescriptor(fir::FirOpBuilder &builder, mlir::Location loc,
                             mlir::Value hostPtr);

/// Generate runtime call to check the section of a descriptor and raise an
/// error if it is not contiguous.
void genDescriptorCheckSection(fir::FirOpBuilder &builder, mlir::Location loc,
                               mlir::Value desc);

} // namespace fir::runtime::cuda

#endif // FORTRAN_OPTIMIZER_BUILDER_RUNTIME_CUDA_DESCRIPTOR_H_

//===- SPIRVOpAvailability.cpp - MLIR SPIR-V Availability Implementation --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Defines the SPIR-V operation availability in the SPIR-V dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/SPIRV/IR/SPIRVOps.h"

// TableGen'erated operation interfaces for querying versions, extensions, and
// capabilities.
#include "mlir/Dialect/SPIRV/IR/SPIRVAvailability.cpp.inc"
#include "mlir/Support/ABINamespace.h"

MLIR_NAMESPACE_BEGIN
namespace spirv {
// TableGen'erated operation availability interface implementations.
#include "mlir/Dialect/SPIRV/IR/SPIRVOpAvailabilityImpl.inc"
}
MLIR_NAMESPACE_END // namespace mlir::spirv

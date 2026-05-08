//===- WaveAMD.h - WaveAMD dialect ------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_WAVE_IR_WAVEAMD_H_
#define MLIR_DIALECT_WAVE_IR_WAVEAMD_H_

#include "mlir/Dialect/Wave/IR/Wave.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "mlir/Dialect/Wave/IR/WaveAMDOpsDialect.h.inc"

#define GET_ATTRDEF_CLASSES
#include "mlir/Dialect/Wave/IR/WaveAMDOpsAttributes.h.inc"

#define GET_TYPEDEF_CLASSES
#include "mlir/Dialect/Wave/IR/WaveAMDOpsTypes.h.inc"

#define GET_OP_CLASSES
#include "mlir/Dialect/Wave/IR/WaveAMDOps.h.inc"

#endif // MLIR_DIALECT_WAVE_IR_WAVEAMD_H_

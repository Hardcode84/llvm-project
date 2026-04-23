//===- Unit.h -  IR Unit definition--------------------*- C++ -*-=============//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_IR_UNIT_H
#define MLIR_IR_UNIT_H

#include "mlir/IR/OperationSupport.h"
#include "llvm/ADT/PointerUnion.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/Compiler.h"
#include "mlir/Support/ABINamespace.h"

LLVM_NAMESPACE_BEGIN
class raw_ostream;
LLVM_NAMESPACE_END // namespace llvm
MLIR_NAMESPACE_BEGIN
class Operation;
class Region;
class Block;
class Value;

/// IRUnit is a union of the different types of IR objects that constitute the
/// IR structure (other than Type and Attribute), that is Operation, Region, and
/// Block.
class IRUnit : public PointerUnion<Operation *, Region *, Block *, Value> {
public:
  using PointerUnion::PointerUnion;

  /// Print the IRUnit to the given stream.
  void print(raw_ostream &os,
             OpPrintingFlags flags =
                 OpPrintingFlags().skipRegions().useLocalScope()) const;
};

raw_ostream &operator<<(raw_ostream &os, const IRUnit &unit);

MLIR_NAMESPACE_END // end namespace mlir

LLVM_NAMESPACE_BEGIN

// Allow llvm::cast style functions.
template <typename To>
struct CastInfo<To, mlir::IRUnit>
    : public CastInfo<To, mlir::IRUnit::PointerUnion> {};

template <typename To>
struct CastInfo<To, const mlir::IRUnit>
    : public CastInfo<To, const mlir::IRUnit::PointerUnion> {};

LLVM_NAMESPACE_END // namespace llvm

#endif // MLIR_IR_UNIT_H

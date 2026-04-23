//===- TypeFromLLVM.h - Translate types from LLVM to MLIR --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the type translation function going from MLIR LLVM dialect
// to LLVM IR and back.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TARGET_LLVMIR_TYPEFROMLLVM_H
#define MLIR_TARGET_LLVMIR_TYPEFROMLLVM_H

#include <memory>
#include "llvm/Support/Compiler.h"
#include "mlir/Support/ABINamespace.h"

LLVM_NAMESPACE_BEGIN
class Type;
LLVM_NAMESPACE_END // namespace llvm

MLIR_NAMESPACE_BEGIN

class Type;
class MLIRContext;

namespace LLVM {

namespace detail {
class TypeFromLLVMIRTranslatorImpl;
} // namespace detail

/// Utility class to translate LLVM IR types to the MLIR LLVM dialect. Stores
/// the translation state, in particular any identified structure types that are
/// reused across translations.
class TypeFromLLVMIRTranslator {
public:
  TypeFromLLVMIRTranslator(MLIRContext &context,
                           bool importStructsAsLiterals = false);
  ~TypeFromLLVMIRTranslator();

  /// Translates the given LLVM IR type to the MLIR LLVM dialect.
  Type translateType(llvm::Type *type);

private:
  /// Private implementation.
  std::unique_ptr<detail::TypeFromLLVMIRTranslatorImpl> impl;
};

} // namespace LLVM
MLIR_NAMESPACE_END // namespace mlir

#endif // MLIR_TARGET_LLVMIR_TYPEFROMLLVM_H

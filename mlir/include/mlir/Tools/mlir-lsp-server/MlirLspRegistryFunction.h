//===- MlirLspRegistryFunction.h - LSP registry functions -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Registry function types for MLIR LSP.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TOOLS_MLIR_LSP_SERVER_MLIRLSPREGISTRYFUNCTION_H
#define MLIR_TOOLS_MLIR_LSP_SERVER_MLIRLSPREGISTRYFUNCTION_H


#include "llvm/Support/Compiler.h"
#include "mlir/Support/ABINamespace.h"
LLVM_NAMESPACE_BEGIN
template <typename Fn>
class function_ref;
namespace lsp {
class URIForFile;
} // namespace lsp
LLVM_NAMESPACE_END // namespace llvm

MLIR_NAMESPACE_BEGIN
class DialectRegistry;
namespace lsp {
using DialectRegistryFn =
    llvm::function_ref<DialectRegistry &(const llvm::lsp::URIForFile &uri)>;
} // namespace lsp
MLIR_NAMESPACE_END // namespace mlir

#endif // MLIR_TOOLS_MLIR_LSP_SERVER_MLIRLSPREGISTRYFUNCTION_H

//===- TableGenLSPServerMain.h - TableGen Language Server main --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Main entry function for tblgen-lsp-server when built as standalone binary.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TOOLS_TBLGENLSPSERVER_TABLEGENLSPSERVERMAIN_H
#define MLIR_TOOLS_TBLGENLSPSERVER_TABLEGENLSPSERVERMAIN_H


#include "llvm/Support/Compiler.h"
#include "mlir/Support/ABINamespace.h"
LLVM_NAMESPACE_BEGIN
struct LogicalResult;
LLVM_NAMESPACE_END // namespace llvm

MLIR_NAMESPACE_BEGIN
/// Implementation for tools like `tblgen-lsp-server`.
llvm::LogicalResult TableGenLspServerMain(int argc, char **argv);
MLIR_NAMESPACE_END // namespace mlir

#endif // MLIR_TOOLS_TBLGENLSPSERVER_TABLEGENLSPSERVERMAIN_H

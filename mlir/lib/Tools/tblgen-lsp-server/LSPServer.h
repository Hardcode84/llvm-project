//===- LSPServer.h - TableGen LSP Server ------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LIB_MLIR_TOOLS_TBLGENLSPSERVER_LSPSERVER_H
#define LIB_MLIR_TOOLS_TBLGENLSPSERVER_LSPSERVER_H

#include <memory>
#include "llvm/Support/Compiler.h"
#include "mlir/Support/ABINamespace.h"

LLVM_NAMESPACE_BEGIN
struct LogicalResult;
namespace lsp {
class JSONTransport;
} // namespace lsp
LLVM_NAMESPACE_END // namespace llvm

MLIR_NAMESPACE_BEGIN
namespace lsp {
class TableGenServer;

/// Run the main loop of the LSP server using the given TableGen server and
/// transport.
llvm::LogicalResult runTableGenLSPServer(TableGenServer &server,
                                         llvm::lsp::JSONTransport &transport);

} // namespace lsp
MLIR_NAMESPACE_END // namespace mlir

#endif // LIB_MLIR_TOOLS_TBLGENLSPSERVER_LSPSERVER_H

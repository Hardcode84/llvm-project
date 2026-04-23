//===- WebAssemblyDisassemblerEmitter.h - Disassembler tables ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file is part of the WebAssembly Disassembler Emitter.
// It contains the interface of the disassembler tables.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_UTILS_TABLEGEN_WEBASSEMBLYDISASSEMBLEREMITTER_H
#define LLVM_UTILS_TABLEGEN_WEBASSEMBLYDISASSEMBLEREMITTER_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Compiler.h"

LLVM_NAMESPACE_BEGIN

class CodeGenInstruction;
class raw_ostream;

void emitWebAssemblyDisassemblerTables(
    raw_ostream &OS, ArrayRef<const CodeGenInstruction *> NumberedInstructions);

LLVM_NAMESPACE_END // namespace llvm

#endif

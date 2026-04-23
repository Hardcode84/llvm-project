//===--------------------- CodeEmitter.cpp ----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the CodeEmitter API.
//
//===----------------------------------------------------------------------===//

#include "llvm/MCA/CodeEmitter.h"
#include "llvm/Support/Compiler.h"

LLVM_NAMESPACE_BEGIN
namespace mca {

CodeEmitter::EncodingInfo CodeEmitter::getOrCreateEncodingInfo(unsigned MCID) {
  EncodingInfo &EI = Encodings[MCID];
  if (EI.second)
    return EI;

  SmallVector<llvm::MCFixup, 2> Fixups;
  const MCInst &Inst = Sequence[MCID];
  EI.first = Code.size();
  MCE.encodeInstruction(Inst, Code, Fixups, STI);
  EI.second = Code.size() - EI.first;
  return EI;
}

} // namespace mca
LLVM_NAMESPACE_END // namespace llvm

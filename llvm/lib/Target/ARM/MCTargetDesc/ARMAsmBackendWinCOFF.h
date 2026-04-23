//===-- ARMAsmBackendWinCOFF.h - ARM Asm Backend WinCOFF --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_ARM_MCTARGETDESC_ARMASMBACKENDWINCOFF_H
#define LLVM_LIB_TARGET_ARM_MCTARGETDESC_ARMASMBACKENDWINCOFF_H

#include "ARMAsmBackend.h"
#include "llvm/MC/MCObjectWriter.h"
#include "llvm/Support/Compiler.h"

LLVM_NAMESPACE_BEGIN
class ARMAsmBackendWinCOFF : public ARMAsmBackend {
public:
  ARMAsmBackendWinCOFF(const Target &T)
      : ARMAsmBackend(T, llvm::endianness::little) {}
  std::unique_ptr<MCObjectTargetWriter>
  createObjectTargetWriter() const override {
    return createARMWinCOFFObjectWriter();
  }
};
LLVM_NAMESPACE_END // namespace llvm

#endif // LLVM_LIB_TARGET_ARM_MCTARGETDESC_ARMASMBACKENDWINCOFF_H

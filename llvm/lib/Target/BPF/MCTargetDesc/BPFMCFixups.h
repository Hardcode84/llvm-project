//=======-- BPFMCFixups.h - BPF-specific fixup entries ------*- C++ -*-=======//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_BPF_MCTARGETDESC_SYSTEMZMCFIXUPS_H
#define LLVM_LIB_TARGET_BPF_MCTARGETDESC_SYSTEMZMCFIXUPS_H

#include "llvm/MC/MCFixup.h"
#include "llvm/Support/Compiler.h"

LLVM_NAMESPACE_BEGIN
namespace BPF {
enum FixupKind {
  // BPF specific relocations.
  FK_BPF_PCRel_4 = FirstTargetFixupKind,

  // Marker
  LastTargetFixupKind,
  NumTargetFixupKinds = LastTargetFixupKind - FirstTargetFixupKind
};
} // end namespace BPF
LLVM_NAMESPACE_END // end namespace llvm

#endif

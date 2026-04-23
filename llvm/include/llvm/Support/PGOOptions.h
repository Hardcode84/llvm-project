//===------ PGOOptions.h -- PGO option tunables ----------------*- C++ -*--===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file
///
/// Define option tunables for PGO.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_SUPPORT_PGOOPTIONS_H
#define LLVM_SUPPORT_PGOOPTIONS_H

#include "llvm/Support/Compiler.h"
#include "llvm/Support/Error.h"

LLVM_NAMESPACE_BEGIN
/// A struct capturing PGO tunables.
struct PGOOptions {
  enum PGOAction { NoAction, IRInstr, IRUse, SampleUse };
  enum CSPGOAction { NoCSAction, CSIRInstr, CSIRUse };
  enum class ColdFuncOpt { Default, OptSize, MinSize, OptNone };
  LLVM_ABI PGOOptions(std::string ProfileFile, std::string CSProfileGenFile,
                      std::string ProfileRemappingFile,
                      std::string MemoryProfile, PGOAction Action = NoAction,
                      CSPGOAction CSAction = NoCSAction,
                      ColdFuncOpt ColdType = ColdFuncOpt::Default,
                      bool DebugInfoForProfiling = false,
                      bool PseudoProbeForProfiling = false,
                      bool AtomicCounterUpdate = false);
  LLVM_ABI PGOOptions(const PGOOptions &);
  LLVM_ABI ~PGOOptions();
  LLVM_ABI PGOOptions &operator=(const PGOOptions &);

  std::string ProfileFile;
  std::string CSProfileGenFile;
  std::string ProfileRemappingFile;
  std::string MemoryProfile;
  PGOAction Action;
  CSPGOAction CSAction;
  ColdFuncOpt ColdOptType;
  bool DebugInfoForProfiling;
  bool PseudoProbeForProfiling;
  bool AtomicCounterUpdate;
};
LLVM_NAMESPACE_END // namespace llvm

#endif

//== ProgramState_Fwd.h - Incomplete declarations of ProgramState -*- C++ -*--=/
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_STATICANALYZER_CORE_PATHSENSITIVE_PROGRAMSTATE_FWD_H
#define LLVM_CLANG_STATICANALYZER_CORE_PATHSENSITIVE_PROGRAMSTATE_FWD_H

#include "clang/Basic/LLVM.h"
#include "llvm/ADT/IntrusiveRefCntPtr.h"
#include "llvm/Support/Compiler.h"

namespace clang {
namespace ento {
  class ProgramState;
  class ProgramStateManager;
  void ProgramStateRetain(const ProgramState *state);
  void ProgramStateRelease(const ProgramState *state);
}
}

LLVM_NAMESPACE_BEGIN
  template <> struct IntrusiveRefCntPtrInfo<const clang::ento::ProgramState> {
    static void retain(const clang::ento::ProgramState *state) {
      clang::ento::ProgramStateRetain(state);
    }
    static void release(const clang::ento::ProgramState *state) {
      clang::ento::ProgramStateRelease(state);
    }
  };
LLVM_NAMESPACE_END

namespace clang {
namespace ento {
  typedef IntrusiveRefCntPtr<const ProgramState> ProgramStateRef;
}
}

#endif


//===- ToolExecutorPluginRegistry.h -----------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLING_TOOLEXECUTORPLUGINREGISTRY_H
#define LLVM_CLANG_TOOLING_TOOLEXECUTORPLUGINREGISTRY_H

#include "clang/Support/Compiler.h"
#include "clang/Tooling/Execution.h"
#include "llvm/Support/Registry.h"
#include "llvm/Support/Compiler.h"

namespace clang {
namespace tooling {

using ToolExecutorPluginRegistry = llvm::Registry<ToolExecutorPlugin>;

} // namespace tooling
} // namespace clang

LLVM_NAMESPACE_BEGIN
extern template class CLANG_TEMPLATE_ABI
    Registry<clang::tooling::ToolExecutorPlugin>;
LLVM_NAMESPACE_END // namespace llvm

#endif // LLVM_CLANG_TOOLING_TOOLEXECUTORPLUGINREGISTRY_H

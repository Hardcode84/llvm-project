//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SOURCE_PLUGINS_EXPRESSIONPARSER_CLANG_INJECTPOINTERSIGNINGFIXUPS_H
#define LLDB_SOURCE_PLUGINS_EXPRESSIONPARSER_CLANG_INJECTPOINTERSIGNINGFIXUPS_H

#include "lldb/lldb-private-enumerations.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/Compiler.h"

LLVM_NAMESPACE_BEGIN
class Function;
class Module;
LLVM_NAMESPACE_END // namespace llvm

namespace lldb_private {

llvm::Error InjectPointerSigningFixupCode(llvm::Module &M,
                                          ExecutionPolicy execution_policy);

} // namespace lldb_private

#endif

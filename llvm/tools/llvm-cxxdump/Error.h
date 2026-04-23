//===- Error.h - system_error extensions for llvm-cxxdump -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This declares a new error_category for the llvm-cxxdump tool.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TOOLS_LLVM_CXXDUMP_ERROR_H
#define LLVM_TOOLS_LLVM_CXXDUMP_ERROR_H

#include <system_error>
#include "llvm/Support/Compiler.h"

LLVM_NAMESPACE_BEGIN
const std::error_category &cxxdump_category();

enum class cxxdump_error {
  success = 0,
  file_not_found,
  unrecognized_file_format,
};

inline std::error_code make_error_code(cxxdump_error e) {
  return std::error_code(static_cast<int>(e), cxxdump_category());
}

LLVM_NAMESPACE_END // namespace llvm

namespace std {
template <>
struct is_error_code_enum<llvm::cxxdump_error> : std::true_type {};
}

#endif

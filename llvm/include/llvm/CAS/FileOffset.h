//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// This file declares interface for FileOffset that represent stored data at an
/// offset from the beginning of a file.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_CAS_FILEOFFSET_H
#define LLVM_CAS_FILEOFFSET_H

#include <cstdint>
#include "llvm/Support/Compiler.h"

LLVM_NAMESPACE_BEGIN
namespace cas {

/// FileOffset is a wrapper around `uint64_t` to represent the offset of data
/// from the beginning of the file.
class FileOffset {
public:
  uint64_t get() const { return Offset; }

  explicit operator bool() const { return Offset; }

  FileOffset() = default;
  explicit FileOffset(uint64_t Offset) : Offset(Offset) {}

private:
  uint64_t Offset = 0;
};

}
LLVM_NAMESPACE_END // namespace llvm::cas

#endif // LLVM_CAS_FILEOFFSET_H

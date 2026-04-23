//===- Archive.h ------------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_OBJCOPY_ARCHIVE_H
#define LLVM_LIB_OBJCOPY_ARCHIVE_H

#include "llvm/Object/ArchiveWriter.h"
#include "llvm/Support/Error.h"
#include <vector>
#include "llvm/Support/Compiler.h"

LLVM_NAMESPACE_BEGIN
namespace objcopy {

class MultiFormatConfig;

/// Applies the transformations described by \p Config to
/// each member in archive \p Ar.
/// \returns Vector of transformed archive members.
Expected<std::vector<NewArchiveMember>>
createNewArchiveMembers(const MultiFormatConfig &Config,
                        const object::Archive &Ar);

} // end namespace objcopy
LLVM_NAMESPACE_END // end namespace llvm

#endif // LLVM_LIB_OBJCOPY_ARCHIVE_H

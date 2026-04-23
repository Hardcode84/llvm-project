//===- ELF AttributeParser.h - ELF Attribute Parser -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SUPPORT_ELFATTRIBUTEPARSER_H
#define LLVM_SUPPORT_ELFATTRIBUTEPARSER_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/Compiler.h"

LLVM_NAMESPACE_BEGIN

class ELFAttributeParser {
public:
  virtual ~ELFAttributeParser() = default;

  virtual Error parse(ArrayRef<uint8_t> Section, llvm::endianness Endian) {
    return llvm::Error::success();
  }
  virtual std::optional<unsigned>
  getAttributeValue(StringRef BuildAttrSubsectionName, unsigned Tag) const {
    return std::nullopt;
  }
  virtual std::optional<unsigned> getAttributeValue(unsigned Tag) const {
    return std::nullopt;
  }
  virtual std::optional<StringRef>
  getAttributeString(StringRef BuildAttrSubsectionName, unsigned Tag) const {
    return std::nullopt;
  }
  virtual std::optional<StringRef> getAttributeString(unsigned Tag) const {
    return std::nullopt;
  }
};

LLVM_NAMESPACE_END // namespace llvm
#endif // LLVM_SUPPORT_ELFATTRIBUTEPARSER_H

//===- LvlTypeParser.h - `LevelType` parser ------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_SPARSETENSOR_IR_DETAIL_LVLTYPEPARSER_H
#define MLIR_DIALECT_SPARSETENSOR_IR_DETAIL_LVLTYPEPARSER_H

#include "mlir/IR/OpImplementation.h"
#include "mlir/Support/ABINamespace.h"

MLIR_NAMESPACE_BEGIN
namespace sparse_tensor {
namespace ir_detail {

class LvlTypeParser {
public:
  LvlTypeParser() = default;
  FailureOr<uint64_t> parseLvlType(AsmParser &parser) const;

private:
  ParseResult parseProperty(AsmParser &parser, uint64_t *properties) const;
  ParseResult parseStructured(AsmParser &parser,
                              SmallVector<unsigned> *structured) const;
};

} // namespace ir_detail
} // namespace sparse_tensor
MLIR_NAMESPACE_END // namespace mlir

#endif // MLIR_DIALECT_SPARSETENSOR_IR_DETAIL_LVLTYPEPARSER_H

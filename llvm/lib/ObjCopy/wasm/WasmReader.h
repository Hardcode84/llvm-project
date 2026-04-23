//===- WasmReader.h ---------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_OBJCOPY_WASM_WASMREADER_H
#define LLVM_LIB_OBJCOPY_WASM_WASMREADER_H

#include "WasmObject.h"
#include "llvm/Support/Compiler.h"

LLVM_NAMESPACE_BEGIN
namespace objcopy {
namespace wasm {

class Reader {
public:
  explicit Reader(const object::WasmObjectFile &O) : WasmObj(O) {}
  Expected<std::unique_ptr<Object>> create() const;

private:
  const object::WasmObjectFile &WasmObj;
};

} // end namespace wasm
} // end namespace objcopy
LLVM_NAMESPACE_END // end namespace llvm

#endif // LLVM_LIB_OBJCOPY_WASM_WASMREADER_H

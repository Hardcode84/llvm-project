//===- TestOpenACC.cpp - OpenACC Test Registration ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains unified registration for all OpenACC test passes.
//
//===----------------------------------------------------------------------===//

#include "mlir/Support/ABINamespace.h"
MLIR_NAMESPACE_BEGIN
namespace test {

// Forward declarations of individual test pass registration functions
void registerTestPointerLikeTypeInterfacePass();
void registerTestRecipePopulatePass();
void registerTestOpenACCSupportPass();

// Unified registration function for all OpenACC tests
void registerTestOpenACC() {
  registerTestPointerLikeTypeInterfacePass();
  registerTestRecipePopulatePass();
  registerTestOpenACCSupportPass();
}

} // namespace test
MLIR_NAMESPACE_END // namespace mlir

//===- unittests/Support/ABINamespaceTestFixture.h - Test fixture --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Shared fixture for ABINamespaceTest.cpp / ABINamespaceTestTU{1,2}.cpp.
// The types must be defined in a single header so that ODR treats them as
// the same type across all three TUs; duplicating the definition in each
// TU would also work, but a header is less error-prone to maintain.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_UNITTESTS_SUPPORT_ABINAMESPACETESTFIXTURE_H
#define LLVM_UNITTESTS_SUPPORT_ABINAMESPACETESTFIXTURE_H

#include "llvm/Support/ABINamespace.h"

LLVM_NAMESPACE_BEGIN
namespace abi_ns_test {

struct Base {
  virtual ~Base() = default;
  virtual int whoami() const { return 1; }
};

struct Derived : Base {
  int whoami() const override { return 2; }
};

} // namespace abi_ns_test
LLVM_NAMESPACE_END

#endif

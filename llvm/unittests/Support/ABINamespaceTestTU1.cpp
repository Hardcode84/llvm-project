//===- unittests/Support/ABINamespaceTestTU1.cpp - cross-TU fixture ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Companion TU for ABINamespaceTest.cpp. Defined in a separate TU on purpose
// so that typeid/dynamic_cast (when RTTI is enabled) and source-level
// resolution of `llvm::abi_ns_test::...` are exercised across TUs.
//
//===----------------------------------------------------------------------===//

#include "ABINamespaceTestFixture.h"

#if defined(__GXX_RTTI) || defined(_CPPRTTI) ||                                \
    (defined(__has_feature) && __has_feature(cxx_rtti))
#define LLVM_ABI_NS_TEST_HAS_RTTI 1
#include <typeinfo>
#else
#define LLVM_ABI_NS_TEST_HAS_RTTI 0
#endif

namespace llvm_abi_ns_testhelpers {

llvm::abi_ns_test::Base *TU1MakeDerived() {
  return new llvm::abi_ns_test::Derived();
}

int TU1CallWhoami(const llvm::abi_ns_test::Base *B) {
  return B->whoami();
}

#if LLVM_ABI_NS_TEST_HAS_RTTI
const std::type_info &TU1BaseTypeID() {
  return typeid(llvm::abi_ns_test::Base);
}

const std::type_info &TU1DerivedTypeID() {
  return typeid(llvm::abi_ns_test::Derived);
}
#endif

} // namespace llvm_abi_ns_testhelpers

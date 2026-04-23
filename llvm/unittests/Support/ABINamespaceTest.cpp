//===- unittests/Support/ABINamespaceTest.cpp - Test the ABI ns macros ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Validates the Stage 1 invariants for LLVM_NAMESPACE_BEGIN / _END.
//
// The core property (cross-TU source-level identity) is always exercised.
// The RTTI-dependent properties (typeid equality, dynamic_cast) are gated
// on RTTI availability since LLVM routinely builds with -fno-rtti.
//
//===----------------------------------------------------------------------===//

#include "ABINamespaceTestFixture.h"
#include "gtest/gtest.h"

#include <cstring>
#include <string>

#if defined(__GXX_RTTI) || defined(_CPPRTTI) ||                                \
    (defined(__has_feature) && __has_feature(cxx_rtti))
#define LLVM_ABI_NS_TEST_HAS_RTTI 1
#include <typeinfo>
#else
#define LLVM_ABI_NS_TEST_HAS_RTTI 0
#endif

// Companion-TU entry points (defined in ABINamespaceTestTU{1,2}.cpp).
namespace llvm_abi_ns_testhelpers {

llvm::abi_ns_test::Base *TU1MakeDerived();
int TU1CallWhoami(const llvm::abi_ns_test::Base *B);
int TU2CallWhoami(const llvm::abi_ns_test::Base *B);

#if LLVM_ABI_NS_TEST_HAS_RTTI
const std::type_info &TU1BaseTypeID();
const std::type_info &TU1DerivedTypeID();
const std::type_info &TU2BaseTypeID();
const std::type_info &TU2DerivedTypeID();
llvm::abi_ns_test::Derived *TU2DynCastToDerived(llvm::abi_ns_test::Base *B);
#endif

} // namespace llvm_abi_ns_testhelpers

namespace {

TEST(ABINamespaceTest, CrossTUSourceLevelIdentity) {
  // Core property. Always runs, regardless of RTTI: an object constructed
  // in TU1 is observable and usable through `llvm::abi_ns_test::Base*` in
  // both this TU and TU2. If the inline-namespace macro expanded to an
  // anonymous namespace (the "defined-but-empty" footgun) this linkage
  // would fail because each TU would see a different type.
  llvm::abi_ns_test::Base *B = llvm_abi_ns_testhelpers::TU1MakeDerived();
  ASSERT_NE(B, nullptr);
  EXPECT_EQ(llvm_abi_ns_testhelpers::TU1CallWhoami(B), 2);
  EXPECT_EQ(llvm_abi_ns_testhelpers::TU2CallWhoami(B), 2);
  delete B;
}

#if LLVM_ABI_NS_TEST_HAS_RTTI
TEST(ABINamespaceTest, CrossTUTypeIDEquality) {
  using llvm_abi_ns_testhelpers::TU1BaseTypeID;
  using llvm_abi_ns_testhelpers::TU1DerivedTypeID;
  using llvm_abi_ns_testhelpers::TU2BaseTypeID;
  using llvm_abi_ns_testhelpers::TU2DerivedTypeID;

  EXPECT_EQ(TU1BaseTypeID(), TU2BaseTypeID());
  EXPECT_EQ(TU1DerivedTypeID(), TU2DerivedTypeID());
  EXPECT_NE(TU1BaseTypeID(), TU1DerivedTypeID());
}

TEST(ABINamespaceTest, CrossTUDynamicCast) {
  using llvm_abi_ns_testhelpers::TU1MakeDerived;
  using llvm_abi_ns_testhelpers::TU2DynCastToDerived;

  llvm::abi_ns_test::Base *B = TU1MakeDerived();
  ASSERT_NE(B, nullptr);
  llvm::abi_ns_test::Derived *D = TU2DynCastToDerived(B);
  ASSERT_NE(D, nullptr);
  EXPECT_EQ(D->whoami(), 2);
  delete B;
}

TEST(ABINamespaceTest, TypeIDNameReflectsTagConfiguration) {
  const char *Name = typeid(llvm::abi_ns_test::Derived).name();
  ASSERT_NE(Name, nullptr);

#ifdef LLVM_ABI_NAMESPACE_ACTIVE
#define LLVM_ABI_NAMESPACE_STR_INNER(x) #x
#define LLVM_ABI_NAMESPACE_STR(x) LLVM_ABI_NAMESPACE_STR_INNER(x)
  const char *Tag = LLVM_ABI_NAMESPACE_STR(LLVM_ABI_NAMESPACE);
  ASSERT_NE(Tag, nullptr);
  ASSERT_GT(std::strlen(Tag), 0u);
  EXPECT_NE(std::string(Name).find(Tag), std::string::npos)
      << "mangled name=" << Name << " should contain tag=" << Tag;
#undef LLVM_ABI_NAMESPACE_STR
#undef LLVM_ABI_NAMESPACE_STR_INNER
#else
  EXPECT_NE(std::string(Name).find("abi_ns_test"), std::string::npos);
  EXPECT_NE(std::string(Name).find("Derived"), std::string::npos);
#endif
}
#endif // LLVM_ABI_NS_TEST_HAS_RTTI

} // namespace

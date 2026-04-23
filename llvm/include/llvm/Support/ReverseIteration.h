#ifndef LLVM_SUPPORT_REVERSEITERATION_H
#define LLVM_SUPPORT_REVERSEITERATION_H

#include "llvm/Config/abi-breaking.h"
#include "llvm/Support/PointerLikeTypeTraits.h"
#include "llvm/Support/Compiler.h"

LLVM_NAMESPACE_BEGIN

template <class T = void *> constexpr bool shouldReverseIterate() {
#if LLVM_ENABLE_REVERSE_ITERATION
  return detail::IsPointerLike<T>::value;
#else
  return false;
#endif
}

LLVM_NAMESPACE_END // namespace llvm
#endif

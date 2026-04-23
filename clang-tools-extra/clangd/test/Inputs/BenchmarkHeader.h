#include "llvm/Support/Compiler.h"
namespace clang {
namespace clangd {
namespace dex {
class Dex;
} // namespace dex
} // namespace clangd
} // namespace clang

LLVM_NAMESPACE_BEGIN
int get_physical_cores();
LLVM_NAMESPACE_END // namespace llvm

namespace {
int Variable;
} // namespace

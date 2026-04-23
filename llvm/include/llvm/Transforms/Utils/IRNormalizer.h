#ifndef LLVM_TRANSFORMS_UTILS_IRNORMALIZER_H
#define LLVM_TRANSFORMS_UTILS_IRNORMALIZER_H

#include "llvm/IR/PassManager.h"
#include "llvm/Support/Compiler.h"

LLVM_NAMESPACE_BEGIN

struct IRNormalizerOptions {
  /// Preserves original instruction order.
  bool PreserveOrder = false;

  /// Renames all instructions (including user-named)
  bool RenameAll = true;

  /// Folds all regular instructions (including pre-outputs)
  bool FoldPreOutputs = true;

  /// Sorts and reorders operands in commutative instructions
  bool ReorderOperands = true;
};

/// IRNormalizer aims to transform LLVM IR into normal form.
struct IRNormalizerPass : public PassInfoMixin<IRNormalizerPass> {
private:
  const IRNormalizerOptions Options;

public:
  IRNormalizerPass(IRNormalizerOptions Options = IRNormalizerOptions())
      : Options(Options) {}

  PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM) const;
};

LLVM_NAMESPACE_END // namespace llvm

#endif // LLVM_TRANSFORMS_UTILS_IRNORMALIZER_H

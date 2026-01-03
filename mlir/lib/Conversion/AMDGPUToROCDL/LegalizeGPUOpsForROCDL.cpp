//===- LegalizeGPUOpsForROCDL.cpp - Legalize GPU ops for ROCDL ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception.
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass to legalize high-level GPU and AMDGPU operations
// by decomposing wide-type values into sequences of i32-based operations.
// This prepares the IR for subsequent conversion to ROCDL.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/AMDGPU/IR/AMDGPUDialect.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
#define GEN_PASS_DEF_LEGALIZEGPUOPSFORROCDLPASS
#include "mlir/Conversion/Passes.h.inc"
} // namespace mlir

using namespace mlir;

// Check if a type needs decomposition to i32 chunks.
static bool needsDecomposition(Type type) {
  // Assume index type is natively supported.
  if (isa<IndexType>(type))
    return false;

  if (auto intType = dyn_cast<IntegerType>(type))
    return intType.getWidth() > 32;
  if (auto floatType = dyn_cast<FloatType>(type))
    return floatType.getWidth() > 32;
  if (auto vectorType = dyn_cast<VectorType>(type)) {
    Type elemType = vectorType.getElementType();
    if (isa<IndexType>(elemType))
      return false;

    unsigned totalBits =
        vectorType.getNumElements() * elemType.getIntOrFloatBitWidth();
    return totalBits > 32;
  }
  return false;
}

// Check if a type needs decomposition for readlane/readfirstlane operations.
// https://llvm.org/docs/AMDGPUUsage.html#llvm-ir-intrinsics
static bool needsBroadcastDecomposition(Type type) {
  // Assume index type is natively supported.
  if (isa<IndexType>(type))
    return false;

  // Supported scalar integer types: i16, i32, i64.
  if (auto intType = dyn_cast<IntegerType>(type)) {
    auto width = static_cast<int>(intType.getWidth());
    return !llvm::is_contained({16, 32, 64}, width);
  }

  // Supported scalar float types: f16, bf16, f32, f64.
  if (isa<Float16Type, BFloat16Type, Float32Type, Float64Type>(type))
    return false;

  // Pointers are supported.
  if (isa<LLVM::LLVMPointerType>(type))
    return false;

  if (auto vecType = dyn_cast<VectorType>(type)) {
    Type elementType = vecType.getElementType();
    // Assume index type is natively supported.
    if (isa<IndexType>(elementType))
      return false;

    // Any vectors of i32 are supported.
    if (elementType.isInteger(32))
      return false;

    // 2x f16/bf16/i16 are supported.
    if (vecType.getNumElements() == 2 &&
        (isa<Float16Type, BFloat16Type>(elementType) ||
         elementType.isInteger(16)))
      return false;
  }

  // All other types need decomposition.
  return true;
}

// Get bit width of a type.
static unsigned getBitWidth(Type type) {
  if (auto intType = dyn_cast<IntegerType>(type))
    return intType.getWidth();
  if (auto floatType = dyn_cast<FloatType>(type))
    return floatType.getWidth();
  if (auto vectorType = dyn_cast<VectorType>(type))
    return vectorType.getNumElements() *
           vectorType.getElementType().getIntOrFloatBitWidth();
  llvm_unreachable("unsupported type");
}

// Decompose a value into dstType chunks.
static SmallVector<Value> decomposeValue(OpBuilder &builder, Location loc,
                                         Value src, Type dstType) {
  Type srcType = src.getType();
  if (srcType == dstType)
    return {src};

  unsigned srcBitWidth = getBitWidth(srcType);
  unsigned dstBitWidth = getBitWidth(dstType);
  if (srcBitWidth == dstBitWidth) {
    Value cast = LLVM::BitcastOp::create(builder, loc, dstType, src);
    return {cast};
  }

  // Handle extension for types narrower than dstType.
  if (dstBitWidth > srcBitWidth) {
    auto smallerInt = builder.getIntegerType(srcBitWidth);
    if (srcType != smallerInt)
      src = LLVM::BitcastOp::create(builder, loc, smallerInt, src);

    auto largerInt = builder.getIntegerType(dstBitWidth);
    Value res = LLVM::ZExtOp::create(builder, loc, largerInt, src);
    return {res};
  }

  assert(srcBitWidth % dstBitWidth == 0 &&
         "src bit width must be a multiple of dst bit width");
  int64_t numElements = srcBitWidth / dstBitWidth;
  auto vecType = VectorType::get(numElements, dstType);

  src = LLVM::BitcastOp::create(builder, loc, vecType, src);

  auto toElements = vector::ToElementsOp::create(builder, loc, src);
  return {toElements.getResults()};
}

// Compose values back into dstType.
static Value composeValue(OpBuilder &builder, Location loc, ValueRange src,
                          Type dstType) {
  assert(!src.empty() && "src range must not be empty");
  if (src.size() == 1) {
    Value res = src.front();
    if (res.getType() == dstType)
      return res;

    // Handle truncation for types narrower than srcType.
    unsigned srcBitWidth = getBitWidth(res.getType());
    unsigned dstBitWidth = getBitWidth(dstType);
    if (dstBitWidth < srcBitWidth) {
      auto largerInt = builder.getIntegerType(srcBitWidth);
      if (res.getType() != largerInt)
        res = LLVM::BitcastOp::create(builder, loc, largerInt, res);

      auto smallerInt = builder.getIntegerType(dstBitWidth);
      res = LLVM::TruncOp::create(builder, loc, smallerInt, res);
    }

    if (res.getType() != dstType)
      res = LLVM::BitcastOp::create(builder, loc, dstType, res);

    return res;
  }

  int64_t numElements = src.size();
  auto srcType = VectorType::get(numElements, src.front().getType());

  Value res = vector::FromElementsOp::create(builder, loc, srcType, src);

  if (res.getType() != dstType)
    res = LLVM::BitcastOp::create(builder, loc, dstType, res);

  return res;
}

//===----------------------------------------------------------------------===//
// Legalization patterns.
//===----------------------------------------------------------------------===//

namespace {

struct LegalizeGPUShufflePattern : public OpRewritePattern<gpu::ShuffleOp> {
  using Base::Base;

  LogicalResult matchAndRewrite(gpu::ShuffleOp op,
                                PatternRewriter &rewriter) const override {
    Value value = op.getValue();
    Type valueType = value.getType();

    if (!needsDecomposition(valueType))
      return failure();

    Location loc = op.getLoc();
    Type i32Type = rewriter.getI32Type();

    SmallVector<Value> inputChunks =
        decomposeValue(rewriter, loc, value, i32Type);

    SmallVector<Value> resultChunks;
    SmallVector<Value> validResults;

    for (Value chunk : inputChunks) {
      auto shuffleResult = gpu::ShuffleOp::create(
          rewriter, loc, chunk, op.getOffset(), op.getWidth(), op.getMode());
      resultChunks.push_back(shuffleResult.getShuffleResult());
      validResults.push_back(shuffleResult.getValid());
    }

    Value composedResult = composeValue(rewriter, loc, resultChunks, valueType);

    // All valid flags should be the same, use the first one.
    rewriter.replaceOp(op, {composedResult, validResults[0]});
    return success();
  }
};

struct LegalizeAMDGPUSwizzleBitModePattern
    : public OpRewritePattern<amdgpu::SwizzleBitModeOp> {
  using Base::Base;

  LogicalResult matchAndRewrite(amdgpu::SwizzleBitModeOp op,
                                PatternRewriter &rewriter) const override {
    Value src = op.getSrc();
    Type srcType = src.getType();

    if (!needsDecomposition(srcType))
      return failure();

    Location loc = op.getLoc();
    Type i32Type = rewriter.getI32Type();

    SmallVector<Value> inputChunks =
        decomposeValue(rewriter, loc, src, i32Type);

    SmallVector<Value> resultChunks;
    for (Value chunk : inputChunks) {
      Value swizzled = amdgpu::SwizzleBitModeOp::create(
          rewriter, loc, chunk, op.getAndMask(), op.getOrMask(),
          op.getXorMask());
      resultChunks.push_back(swizzled);
    }

    Value composedResult = composeValue(rewriter, loc, resultChunks, srcType);
    rewriter.replaceOp(op, composedResult);
    return success();
  }
};

struct LegalizeAMDGPUPermlaneSwapPattern
    : public OpRewritePattern<amdgpu::PermlaneSwapOp> {
  using Base::Base;

  LogicalResult matchAndRewrite(amdgpu::PermlaneSwapOp op,
                                PatternRewriter &rewriter) const override {
    Value src = op.getSrc();
    Type srcType = src.getType();

    if (!needsDecomposition(srcType))
      return failure();

    Location loc = op.getLoc();
    Type i32Type = rewriter.getI32Type();

    SmallVector<Value> inputChunks =
        decomposeValue(rewriter, loc, src, i32Type);

    SmallVector<Value> resultChunks;
    for (Value chunk : inputChunks) {
      Value permuted = amdgpu::PermlaneSwapOp::create(
          rewriter, loc, chunk, op.getRowLength(), op.getFetchInactive(),
          op.getBoundCtrl());
      resultChunks.push_back(permuted);
    }

    Value composedResult = composeValue(rewriter, loc, resultChunks, srcType);
    rewriter.replaceOp(op, composedResult);
    return success();
  }
};

struct LegalizeGPUSubgroupBroadcastPattern
    : public OpRewritePattern<gpu::SubgroupBroadcastOp> {
  using Base::Base;

  LogicalResult matchAndRewrite(gpu::SubgroupBroadcastOp op,
                                PatternRewriter &rewriter) const override {
    Value src = op.getSrc();
    Type srcType = src.getType();

    // Only decompose if needed for broadcast operations.
    if (!needsBroadcastDecomposition(srcType))
      return failure();

    Location loc = op.getLoc();
    Type i32Type = rewriter.getI32Type();

    // Decompose input value.
    SmallVector<Value> inputChunks =
        decomposeValue(rewriter, loc, src, i32Type);

    // Apply broadcast to each chunk.
    SmallVector<Value> resultChunks;
    for (Value chunk : inputChunks) {
      Value broadcasted = gpu::SubgroupBroadcastOp::create(
          rewriter, loc, chunk, op.getLane(), op.getBroadcastType());
      resultChunks.push_back(broadcasted);
    }

    // Compose result chunks.
    Value composedResult = composeValue(rewriter, loc, resultChunks, srcType);
    rewriter.replaceOp(op, composedResult);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Pass implementation.
//===----------------------------------------------------------------------===//

struct LegalizeGPUOpsForROCDLPass
    : public impl::LegalizeGPUOpsForROCDLPassBase<LegalizeGPUOpsForROCDLPass> {
  using Base::Base;

  void runOnOperation() override {
    Operation *op = getOperation();
    MLIRContext *ctx = op->getContext();

    RewritePatternSet patterns(ctx);
    patterns.add<LegalizeGPUShufflePattern, LegalizeGPUSubgroupBroadcastPattern,
                 LegalizeAMDGPUSwizzleBitModePattern,
                 LegalizeAMDGPUPermlaneSwapPattern>(ctx);

    if (failed(applyPatternsGreedily(op, std::move(patterns))))
      return signalPassFailure();
  }
};

} // namespace

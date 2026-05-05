//===- WaveToROCDL.cpp - Lower Wave to ROCDL/SCF dialects -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Wave/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Wave/IR/Wave.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::wave {
#define GEN_PASS_DEF_CONVERTWAVETOROCDL
#include "mlir/Dialect/Wave/Transforms/Passes.h.inc"
} // namespace mlir::wave

using namespace mlir;
using namespace mlir::wave;

namespace {

static Value createI32Constant(PatternRewriter &rewriter, Location loc,
                               int32_t value) {
  return arith::ConstantIntOp::create(rewriter, loc, value, /*width=*/32);
}

static Value castI32ToIndex(PatternRewriter &rewriter, Location loc,
                            Type resultType, Value value) {
  if (resultType.isIndex())
    return arith::IndexCastOp::create(rewriter, loc, resultType, value);
  return value;
}

static Type getLoweredType(Type type) {
  if (auto simd = dyn_cast<SimdType>(type))
    return simd.getElementType();
  if (auto mask = dyn_cast<MaskType>(type))
    return IntegerType::get(type.getContext(), 1);
  return type;
}

struct LaneIdLowering : OpRewritePattern<LaneIdOp> {
  using OpRewritePattern<LaneIdOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(LaneIdOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Type i32 = rewriter.getI32Type();
    Value minusOne = createI32Constant(rewriter, loc, -1);
    Value zero = createI32Constant(rewriter, loc, 0);
    auto emptyArray = rewriter.getArrayAttr({});
    Value lane = ROCDL::MbcntLoOp::create(rewriter, loc, i32, minusOne, zero,
                                          emptyArray, emptyArray);
    Type resultType = getLoweredType(op.getType());
    if (resultType.isIndex())
      lane = castI32ToIndex(rewriter, loc, resultType, lane);
    rewriter.replaceOp(op, lane);
    return success();
  }
};

struct SubgroupIdLowering : OpRewritePattern<SubgroupIdOp> {
  using OpRewritePattern<SubgroupIdOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(SubgroupIdOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value subgroup =
        ROCDL::WaveId::create(rewriter, loc, rewriter.getI32Type());
    rewriter.replaceOp(op,
                       castI32ToIndex(rewriter, loc, op.getType(), subgroup));
    return success();
  }
};

struct SubgroupSizeLowering : OpRewritePattern<SubgroupSizeOp> {
  using OpRewritePattern<SubgroupSizeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(SubgroupSizeOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value size =
        ROCDL::WavefrontSizeOp::create(rewriter, loc, rewriter.getI32Type());
    rewriter.replaceOp(op, castI32ToIndex(rewriter, loc, op.getType(), size));
    return success();
  }
};

struct BallotLowering : OpRewritePattern<BallotOp> {
  using OpRewritePattern<BallotOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(BallotOp op,
                                PatternRewriter &rewriter) const override {
    if (isa<MaskType>(op.getMask().getType()))
      return failure();
    rewriter.replaceOpWithNewOp<ROCDL::BallotOp>(op, op.getType(),
                                                 op.getMask());
    return success();
  }
};

struct SplatLowering : OpRewritePattern<SplatOp> {
  using OpRewritePattern<SplatOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(SplatOp op,
                                PatternRewriter &rewriter) const override {
    rewriter.replaceOp(op, op.getSource());
    return success();
  }
};

struct BinaryLowering : OpRewritePattern<BinaryOp> {
  using OpRewritePattern<BinaryOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(BinaryOp op,
                                PatternRewriter &rewriter) const override {
    if (isa<SimdType>(op.getLhs().getType()) ||
        isa<SimdType>(op.getRhs().getType()))
      return failure();
    StringRef kind = op.getKind();
    Value result;
    if (kind == "addi")
      result = arith::AddIOp::create(rewriter, op.getLoc(), op.getLhs(),
                                     op.getRhs());
    else if (kind == "andi")
      result = arith::AndIOp::create(rewriter, op.getLoc(), op.getLhs(),
                                     op.getRhs());
    else if (kind == "ori")
      result =
          arith::OrIOp::create(rewriter, op.getLoc(), op.getLhs(), op.getRhs());
    else if (kind == "xori")
      result = arith::XOrIOp::create(rewriter, op.getLoc(), op.getLhs(),
                                     op.getRhs());
    else if (kind == "shli")
      result = arith::ShLIOp::create(rewriter, op.getLoc(), op.getLhs(),
                                     op.getRhs());
    else
      return rewriter.notifyMatchFailure(op, "unsupported wave binary kind");
    rewriter.replaceOp(op, result);
    return success();
  }
};

struct CmpILowering : OpRewritePattern<CmpIOp> {
  using OpRewritePattern<CmpIOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(CmpIOp op,
                                PatternRewriter &rewriter) const override {
    if (isa<SimdType>(op.getLhs().getType()) ||
        isa<SimdType>(op.getRhs().getType()))
      return failure();
    Value result = arith::CmpIOp::create(rewriter, op.getLoc(),
                                         op.getPredicate(), op.getLhs(),
                                         op.getRhs());
    rewriter.replaceOp(op, result);
    return success();
  }
};

struct ReadFirstLowering : OpRewritePattern<ReadFirstOp> {
  using OpRewritePattern<ReadFirstOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ReadFirstOp op,
                                PatternRewriter &rewriter) const override {
    if (isa<SimdType>(op.getSource().getType()))
      return failure();
    rewriter.replaceOpWithNewOp<ROCDL::ReadfirstlaneOp>(
        op, op.getType(), op.getSource());
    return success();
  }
};

static LogicalResult replaceWaveYieldWithScfYield(Region &region,
                                                  PatternRewriter &rewriter) {
  if (region.empty())
    return success();
  Operation *terminator = region.front().getTerminator();
  auto yield = dyn_cast<YieldOp>(terminator);
  if (!yield)
    return failure();
  rewriter.setInsertionPoint(yield);
  rewriter.replaceOpWithNewOp<scf::YieldOp>(yield);
  return success();
}

struct WhereLowering : OpRewritePattern<WhereOp> {
  using OpRewritePattern<WhereOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(WhereOp op,
                                PatternRewriter &rewriter) const override {
    if (isa<MaskType>(op.getCondition().getType()))
      return failure();
    OperationState state(op.getLoc(), scf::IfOp::getOperationName());
    scf::IfOp::build(rewriter, state, TypeRange{}, op.getCondition(),
                     /*addThenBlock=*/false, /*addElseBlock=*/false);
    Operation *newOp = rewriter.create(state);
    auto ifOp = cast<scf::IfOp>(newOp);

    ifOp.getThenRegion().takeBody(op.getThenRegion());
    if (!op.getElseRegion().empty())
      ifOp.getElseRegion().takeBody(op.getElseRegion());

    if (failed(replaceWaveYieldWithScfYield(ifOp.getThenRegion(), rewriter)) ||
        failed(replaceWaveYieldWithScfYield(ifOp.getElseRegion(), rewriter)))
      return failure();

    rewriter.eraseOp(op);
    return success();
  }
};

struct ConvertWaveToROCDLPass
    : public wave::impl::ConvertWaveToROCDLBase<ConvertWaveToROCDLPass> {
  void runOnOperation() override {
    RewritePatternSet valuePatterns(&getContext());
    valuePatterns.add<LaneIdLowering, SubgroupIdLowering, SubgroupSizeLowering,
                      SplatLowering>(&getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(valuePatterns))))
      return signalPassFailure();

    RewritePatternSet computePatterns(&getContext());
    computePatterns.add<BinaryLowering, CmpILowering>(&getContext());
    if (failed(
            applyPatternsGreedily(getOperation(), std::move(computePatterns))))
      return signalPassFailure();

    RewritePatternSet boundaryPatterns(&getContext());
    boundaryPatterns
        .add<BallotLowering, ReadFirstLowering, WhereLowering>(
            &getContext());
    if (failed(applyPatternsGreedily(getOperation(),
                                     std::move(boundaryPatterns))))
      signalPassFailure();
  }
};

} // namespace

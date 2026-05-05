//===-- WaveOps.td - Wave dialect operations --------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_WAVE_IR_WAVEOPS_TD
#define MLIR_DIALECT_WAVE_IR_WAVEOPS_TD

include "mlir/Dialect/Wave/IR/WaveBase.td"
include "mlir/Dialect/Wave/IR/WaveTypes.td"

include "mlir/Dialect/Arith/IR/ArithBase.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

class Wave_Op<string mnemonic, list<Trait> traits = []> :
    Op<Wave_Dialect, mnemonic, traits>;

def Wave_YieldOp : Wave_Op<"yield", [Pure, ReturnLike, Terminator,
                                     ParentOneOf<["WhereOp"]>]> {
  let summary = "terminator for wave structured regions";
  let description = [{
    Terminates a Wave dialect structured region.

    The MVP Wave dialect currently uses `wave.yield` only for side-effecting
    `wave.where` regions with no yielded values.
  }];

  let assemblyFormat = "attr-dict";
}

def Wave_LaneIdOp : Wave_Op<"lane_id", [Pure]> {
  let summary = "lane id within the current subgroup";
  let description = [{
    Returns the lane id as a lane-varying SIMD value. Each lane observes its own
    id in the subgroup/wave.
  }];

  let results = (outs Wave_Simd:$result);
  let assemblyFormat = "attr-dict `:` qualified(type($result))";
}

def Wave_SubgroupIdOp : Wave_Op<"subgroup_id", [Pure]> {
  let summary = "subgroup id within the current workgroup";
  let description = [{
    Returns the subgroup/wave id within the current workgroup.
  }];

  let results = (outs Index:$result);
  let assemblyFormat = "attr-dict";
}

def Wave_SubgroupSizeOp : Wave_Op<"subgroup_size", [Pure]> {
  let summary = "number of lanes in the current subgroup";
  let description = [{
    Returns the number of lanes in the current subgroup/wave.
  }];

  let results = (outs Index:$result);
  let assemblyFormat = "attr-dict";
}

def Wave_BallotOp : Wave_Op<"ballot", [NoMemoryEffect]> {
  let summary = "materialize a logical mask as integer bits";
  let description = [{
    Materializes a first-class `!wave.mask<W>` as an integer bit mask whose bit
    `N` corresponds to lane `N`.

    The result width is target-dependent. AMDGPU wave32 commonly uses `i32`;
    wave64 commonly uses `i64`.
  }];

  let arguments = (ins Wave_Mask:$mask);
  let results = (outs AnySignlessInteger:$result);
  let assemblyFormat = "$mask attr-dict `:` qualified(type($mask)) `->` type($result)";
  let hasVerifier = 1;
}

def Wave_SplatOp : Wave_Op<"splat", [Pure]> {
  let summary = "broadcast a scalar value into a lane-varying SIMD value";
  let description = [{
    Broadcasts an ordinary scalar value into a `!wave.simd<T, W>`. Lowering to
    AMDGPU can keep the original scalar value as an SGPR operand when legal.
  }];

  let arguments = (ins AnyType:$source);
  let results = (outs Wave_Simd:$result);
  let assemblyFormat = "$source attr-dict `:` type($source) `->` qualified(type($result))";
  let hasVerifier = 1;
}

def Wave_BinaryOp : Wave_Op<"binary", [Pure]> {
  let summary = "lane-wise integer binary operation";
  let description = [{
    Applies an integer binary operation lane-wise to two SIMD values.
  }];

  let arguments = (ins StrAttr:$kind, Wave_Simd:$lhs, Wave_Simd:$rhs);
  let results = (outs Wave_Simd:$result);
  let assemblyFormat = "$kind $lhs `,` $rhs attr-dict `:` qualified(type($lhs)) `,` qualified(type($rhs)) `->` qualified(type($result))";
  let hasVerifier = 1;
}

def Wave_CmpIOp : Wave_Op<"cmpi", [Pure]> {
  let summary = "lane-wise integer comparison";
  let description = [{
    Compares two SIMD integer values lane-wise and returns a first-class
    `!wave.mask<W>`.
  }];

  let arguments = (ins Arith_CmpIPredicateAttr:$predicate,
                       Wave_Simd:$lhs,
                       Wave_Simd:$rhs);
  let results = (outs Wave_Mask:$result);
  let assemblyFormat = "$predicate $lhs `,` $rhs attr-dict `:` qualified(type($lhs)) `,` qualified(type($rhs)) `->` qualified(type($result))";
  let hasVerifier = 1;
}

def Wave_ReadFirstOp : Wave_Op<"read_first", [
    NoMemoryEffect]> {
  let summary = "broadcast the first active lane's value";
  let description = [{
    Returns the value from the first active lane, broadcast to all active lanes.
    This is the explicit lane-varying to scalar-uniform boundary.
  }];

  let arguments = (ins Wave_Simd:$source);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$source attr-dict `:` qualified(type($source)) `->` type($result)";
  let hasVerifier = 1;
}

def Wave_TokenOp : Wave_Op<"token", [Pure]> {
  let summary = "create an empty memory dependency token";
  let results = (outs Wave_MemToken:$result);
  let assemblyFormat = "attr-dict `:` qualified(type($result))";
}

def Wave_AfterOp : Wave_Op<"after", [Pure]> {
  let summary = "derive a memory token after one or more dependencies";
  let arguments = (ins Variadic<Wave_MemToken>:$dependencies);
  let results = (outs Wave_MemToken:$result);
  let assemblyFormat = "$dependencies attr-dict `:` type($dependencies) `->` qualified(type($result))";
}

def Wave_JoinOp : Wave_Op<"join", [Pure]> {
  let summary = "join multiple memory dependency tokens";
  let arguments = (ins Variadic<Wave_MemToken>:$dependencies);
  let results = (outs Wave_MemToken:$result);
  let assemblyFormat = "$dependencies attr-dict `:` type($dependencies) `->` qualified(type($result))";
}

def Wave_WaitOp : Wave_Op<"wait", []> {
  let summary = "wait for explicit memory dependencies";
  let arguments = (ins Variadic<Wave_MemToken>:$dependencies);
  let assemblyFormat = "$dependencies attr-dict `:` type($dependencies)";
}

def Wave_PtrAddOp : Wave_Op<"ptr_add", [Pure]> {
  let summary = "add an element offset to a Wave pointer";
  let arguments = (ins AnyType:$base, AnyType:$offset);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$base `,` $offset attr-dict `:` type($base) `,` type($offset) `->` type($result)";
  let hasVerifier = 1;
}

def Wave_StoreOp : Wave_Op<"store", [MemoryEffects<[MemWrite]>]> {
  let summary = "store a lane-varying SIMD value to memory";
  let description = [{
    Stores a value to a `!wave.ptr<T, space>` or lane-varying
    `!wave.simd<!wave.ptr<T, space>, W>` pointer.
  }];

  let arguments = (ins Wave_Simd:$value,
                       AnyType:$ptr,
                       Optional<Wave_MemToken>:$dependency);
  let results = (outs Wave_MemToken:$token);
  let assemblyFormat = "$value `->` $ptr (`after` $dependency^)? attr-dict `:` functional-type(operands, results)";
  let hasVerifier = 1;
}

def Wave_WhereOp : Wave_Op<"where", [
    SingleBlockImplicitTerminator<"YieldOp">,
    RecursiveMemoryEffects,
    NoRegionArguments]> {
  let summary = "structured lane-mask predication";
  let description = [{
    Executes the `then` region under a lane-varying mask. If an `otherwise`
    region is present, it executes under the complementary mask within the
    original active lanes.

    In the MVP lowering, this operation is converted to `scf.if` and relies on
    target lowering to preserve divergent-control semantics. Later target-specific
    lowerings can consume `wave.where` directly and lower it to EXEC/predicate
    manipulation without reconstructing structure from generic control flow.
  }];

  let arguments = (ins Wave_Mask:$condition);
  let regions = (region SizedRegion<1>:$thenRegion,
                        MaxSizedRegion<1>:$elseRegion);

  let assemblyFormat = [{
    $condition $thenRegion (`otherwise` $elseRegion^)? attr-dict `:` qualified(type($condition))
  }];

  let hasVerifier = 1;
}

#endif // MLIR_DIALECT_WAVE_IR_WAVEOPS_TD

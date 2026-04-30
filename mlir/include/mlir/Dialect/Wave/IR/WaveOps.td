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
    Returns the current lane id within the subgroup/wave. The result is an
    `index` value so it can be used naturally with `scf` and `memref` indices.
  }];

  let results = (outs Index:$result);
  let assemblyFormat = "attr-dict";
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
  let summary = "collect a lane predicate into a logical mask";
  let description = [{
    Performs a subgroup ballot. Each lane contributes its `i1` predicate value
    and the result is an integer mask whose bit `N` corresponds to lane `N`.

    The result width is target-dependent. AMDGPU wave32 commonly uses `i32`;
    wave64 commonly uses `i64`.
  }];

  let arguments = (ins I1:$predicate);
  let results = (outs AnySignlessInteger:$result);
  let assemblyFormat = "$predicate attr-dict `:` type($result)";
}

def Wave_ReadFirstOp : Wave_Op<"read_first", [
    NoMemoryEffect, AllTypesMatch<["source", "result"]>]> {
  let summary = "broadcast the first active lane's value";
  let description = [{
    Returns the value from the first active lane, broadcast to all active lanes.
    This is the explicit lane-varying to scalar-uniform boundary.
  }];

  let arguments = (ins AnyType:$source);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$source attr-dict `:` type($result)";
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

  let arguments = (ins I1:$condition);
  let regions = (region SizedRegion<1>:$thenRegion,
                        MaxSizedRegion<1>:$elseRegion);

  let assemblyFormat = [{
    $condition $thenRegion (`otherwise` $elseRegion^)? attr-dict
  }];

  let hasVerifier = 1;
}

#endif // MLIR_DIALECT_WAVE_IR_WAVEOPS_TD

//===-- WaveAMDOps.td - WaveAMD dialect operations --------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_WAVE_IR_WAVEAMDOPS_TD
#define MLIR_DIALECT_WAVE_IR_WAVEAMDOPS_TD

include "mlir/Dialect/Wave/IR/WaveAMDBase.td"
include "mlir/Dialect/Wave/IR/WaveAMDTypes.td"
include "mlir/Dialect/Wave/IR/WaveTypes.td"

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

class WaveAMD_Op<string mnemonic, list<Trait> traits = []> :
    Op<WaveAMD_Dialect, mnemonic, traits>;

def WaveAMD_MakeBufferOp : WaveAMD_Op<"make_buffer", [Pure]> {
  let summary = "construct an AMD buffer pointer from a flat/global pointer";
  let description = [{
    Constructs a `!wave.ptr<T, #waveamd.buffer>` value from a uniform base
    pointer and a uniform byte range. The resulting pointer is represented by a
    uniform AMD buffer descriptor plus the lane-varying i32 offset carried by
    subsequent `wave.ptr_add` operations.
  }];
  let arguments = (ins Wave_Ptr:$base,
                       AnySignlessInteger:$range);
  let results = (outs Wave_Ptr:$result);
  let assemblyFormat = "$base `,` $range attr-dict `:` type($base) `,` type($range) `->` type($result)";
  let hasVerifier = 1;
}

def WaveAMD_FragmentFillOp : WaveAMD_Op<"fragment_fill", [Pure]> {
  let summary = "fill an AMD matrix fragment with a scalar bit pattern";
  let arguments = (ins AnySignlessInteger:$source);
  let results = (outs WaveAMD_Fragment:$result);
  let assemblyFormat = "$source attr-dict `:` type($source) `->` qualified(type($result))";
  let hasVerifier = 1;
}

def WaveAMD_MmaOp : WaveAMD_Op<"mma", [Pure]> {
  let summary = "AMD wave-cooperative matrix multiply-accumulate";
  let arguments = (ins StrAttr:$kind,
                       WaveAMD_Fragment:$a,
                       WaveAMD_Fragment:$b,
                       WaveAMD_Fragment:$acc);
  let results = (outs WaveAMD_Fragment:$result);
  let assemblyFormat = "$kind $a `,` $b `,` $acc attr-dict `:` qualified(type($a)) `,` qualified(type($b)) `,` qualified(type($acc)) `->` qualified(type($result))";
  let hasVerifier = 1;
}

def WaveAMD_FragmentStoreOp : WaveAMD_Op<"fragment_store", [MemoryEffects<[MemWrite]>]> {
  let summary = "store an AMD matrix fragment to memory";
  let arguments = (ins WaveAMD_Fragment:$fragment,
                       AnyType:$ptr,
                       Optional<Wave_MemToken>:$dependency);
  let results = (outs Wave_MemToken:$token);
  let assemblyFormat = "$fragment `->` $ptr (`after` $dependency^)? attr-dict `:` functional-type(operands, results)";
  let hasVerifier = 1;
}

#endif // MLIR_DIALECT_WAVE_IR_WAVEAMDOPS_TD

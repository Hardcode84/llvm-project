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

include "mlir/Dialect/MemRef/IR/MemRefBase.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

class WaveAMD_Op<string mnemonic, list<Trait> traits = []> :
    Op<WaveAMD_Dialect, mnemonic, traits>;

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

def WaveAMD_FragmentStoreOp : WaveAMD_Op<"fragment_store", [
    MemoryEffects<[MemWrite]>, AttrSizedOperandSegments]> {
  let summary = "store an AMD matrix fragment to memory";
  let arguments = (ins WaveAMD_Fragment:$fragment,
                       AnyMemRef:$memref,
                       Variadic<AnyType>:$indices,
                       Optional<Wave_MemToken>:$dependency);
  let results = (outs Wave_MemToken:$token);
  let assemblyFormat = "$fragment `->` $memref `[` $indices `]` (`after` $dependency^)? attr-dict `:` functional-type(operands, results)";
  let hasVerifier = 1;
}

#endif // MLIR_DIALECT_WAVE_IR_WAVEAMDOPS_TD

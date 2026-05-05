//===-- WaveAMDTypes.td - WaveAMD dialect types -----------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_WAVE_IR_WAVEAMDTYPES_TD
#define MLIR_DIALECT_WAVE_IR_WAVEAMDTYPES_TD

include "mlir/Dialect/Wave/IR/WaveAMDBase.td"
include "mlir/IR/AttrTypeBase.td"

class WaveAMD_Type<string name, string typeMnemonic, list<Trait> traits = [],
                   string baseCppClass = "::mlir::Type">
    : TypeDef<WaveAMD_Dialect, name, traits, baseCppClass> {
  let mnemonic = typeMnemonic;
}

def WaveAMD_FragmentType : WaveAMD_Type<"Fragment", "fragment"> {
  let summary = "AMD wave-cooperative matrix fragment";
  let description = [{
    A `!waveamd.fragment<role, T, M, N, W, R>` value is an AMD-specific
    distributed matrix fragment. `role` is 0 for an A operand, 1 for a B
    operand, and 2 for an accumulator/result. `T`, `M`, and `N` describe the
    logical tile, `W` is the wave width, and `R` is the number of 32-bit
    registers used by each lane.
  }];

  let parameters = (ins "int64_t":$role,
                        "mlir::Type":$elementType,
                        "int64_t":$rows,
                        "int64_t":$columns,
                        "int64_t":$waveSize,
                        "int64_t":$registers);
  let assemblyFormat =
      "`<` $role `,` $elementType `,` $rows `,` $columns `,` $waveSize `,` $registers `>`";
}

def WaveAMD_Fragment
    : Type<CPred<"::llvm::isa<::mlir::waveamd::FragmentType>($_self)">,
           "WaveAMD matrix fragment type">;

#endif // MLIR_DIALECT_WAVE_IR_WAVEAMDTYPES_TD

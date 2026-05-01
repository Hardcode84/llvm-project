//===-- WaveTypes.td - Wave dialect types -----------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_WAVE_IR_WAVETYPES_TD
#define MLIR_DIALECT_WAVE_IR_WAVETYPES_TD

include "mlir/Dialect/Wave/IR/WaveBase.td"
include "mlir/IR/AttrTypeBase.td"

class Wave_Type<string name, string typeMnemonic, list<Trait> traits = [],
                string baseCppClass = "::mlir::Type">
    : TypeDef<Wave_Dialect, name, traits, baseCppClass> {
  let mnemonic = typeMnemonic;
}

def Wave_SimdType : Wave_Type<"Simd", "simd"> {
  let summary = "lane-varying SIMD value";
  let description = [{
    A `!wave.simd<T, W>` value is one logical `T` value per active lane of a
    subgroup with width `W`. Lowering to AMDGPU erases the type to the element
    type `T`; the value is then represented by the usual divergent VGPR
    semantics in the AMDGPU backend.
  }];

  let parameters = (ins "mlir::Type":$elementType, "int64_t":$width);
  let assemblyFormat = "`<` $elementType `,` $width `>`";
}

def Wave_MaskType : Wave_Type<"Mask", "mask"> {
  let summary = "logical lane mask";
  let description = [{
    A `!wave.mask<W>` value is one logical predicate bit per lane of a subgroup
    with width `W`. Lowering may choose a divergent `i1`, AMDGPU EXEC-width
    scalar mask, target predicate register, or another physical form.
  }];

  let parameters = (ins "int64_t":$width);
  let assemblyFormat = "`<` $width `>`";
}

def Wave_FragmentType : Wave_Type<"Fragment", "fragment"> {
  let summary = "wave-cooperative matrix fragment";
  let description = [{
    A `!wave.fragment<role, T, M, N, W, R>` value is a distributed matrix
    fragment. `role` is 0 for an A operand, 1 for a B operand, and 2 for an
    accumulator/result. `T`, `M`, and `N` describe the logical tile, `W` is the
    wave width, and `R` is the number of 32-bit registers used by each lane.
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

def Wave_Simd : Type<CPred<"::llvm::isa<::mlir::wave::SimdType>($_self)">,
                     "wave SIMD type">;

def Wave_Mask : Type<CPred<"::llvm::isa<::mlir::wave::MaskType>($_self)">,
                     "wave mask type">;

def Wave_Fragment
    : Type<CPred<"::llvm::isa<::mlir::wave::FragmentType>($_self)">,
           "wave matrix fragment type">;

#endif // MLIR_DIALECT_WAVE_IR_WAVETYPES_TD

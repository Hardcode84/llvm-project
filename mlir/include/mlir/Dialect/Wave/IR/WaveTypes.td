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
include "mlir/Dialect/Wave/IR/WaveAttrs.td"
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

def Wave_MemTokenType : Wave_Type<"MemToken", "mem.token"> {
  let summary = "explicit memory dependency token";
  let description = [{
    A `!wave.mem.token` value represents an explicit ordering dependency
    between Wave memory operations. Memory operations that exchange tokens are
    ordered; operations with no token relationship may be scheduled
    independently.
  }];
}

def Wave_PtrType : Wave_Type<"Ptr", "ptr"> {
  let summary = "Wave pointer";
  let description = [{
    A `!wave.ptr<T, #space>` value is a pointer to an element of type `T` in an
    arbitrary address space attribute. The `wave` dialect defines
    `#wave.global`, `#wave.shared`, and `#wave.private`; target-specific
    dialects may provide additional address-space attributes.
  }];

  let parameters = (ins "mlir::Type":$elementType,
                        "mlir::Attribute":$addressSpace);
  let assemblyFormat = "`<` $elementType `,` $addressSpace `>`";
}

def Wave_Simd : Type<CPred<"::llvm::isa<::mlir::wave::SimdType>($_self)">,
                     "wave SIMD type">;

def Wave_Mask : Type<CPred<"::llvm::isa<::mlir::wave::MaskType>($_self)">,
                     "wave mask type">;

def Wave_MemToken
    : Type<CPred<"::llvm::isa<::mlir::wave::MemTokenType>($_self)">,
           "wave memory token type">;

def Wave_Ptr : Type<CPred<"::llvm::isa<::mlir::wave::PtrType>($_self)">,
                    "wave pointer type">;

#endif // MLIR_DIALECT_WAVE_IR_WAVETYPES_TD

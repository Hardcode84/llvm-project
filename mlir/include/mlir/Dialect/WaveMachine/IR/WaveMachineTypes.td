//===-- WaveMachineTypes.td - WaveMachine dialect types -----*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_WAVEMACHINE_IR_WAVEMACHINETYPES_TD
#define MLIR_DIALECT_WAVEMACHINE_IR_WAVEMACHINETYPES_TD

include "mlir/Dialect/WaveMachine/IR/WaveMachineBase.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/EnumAttr.td"

class WaveMachine_Type<string name, string typeMnemonic,
                       list<Trait> traits = [],
                       string baseCppClass = "::mlir::Type">
    : TypeDef<WaveMachine_Dialect, name, traits, baseCppClass> {
  let mnemonic = typeMnemonic;
}

def WaveMachine_RegClassSGPR : I32EnumAttrCase<"SGPR", 0, "sgpr">;
def WaveMachine_RegClassVGPR : I32EnumAttrCase<"VGPR", 1, "vgpr">;
def WaveMachine_RegClassAGPR : I32EnumAttrCase<"AGPR", 2, "agpr">;

def WaveMachine_RegClassEnum : I32EnumAttr<"RegClass",
    "WaveMachine register class",
    [WaveMachine_RegClassSGPR, WaveMachine_RegClassVGPR,
     WaveMachine_RegClassAGPR]> {
  let cppNamespace = "::mlir::wavemachine";
}

def WaveMachine_RegType : WaveMachine_Type<"Reg", "reg"> {
  let summary = "machine register value";
  let description = [{
    A `!wavemachine.reg<class, width[, index]>` value represents a selected
    virtual or physical AMDGPU register. The `class` parameter is an enum
    (`sgpr`, `vgpr`, or `agpr`), `width` is the number of consecutive 32-bit
    registers, and optional `index` is the assigned physical register base.
  }];

  let parameters = (ins EnumParameter<WaveMachine_RegClassEnum>:$regClass,
                        "int64_t":$width,
                        DefaultValuedParameter<"int64_t", "-1">:$index);
  let assemblyFormat = "`<` $regClass `,` $width (`,` $index^)? `>`";
}

def WaveMachine_ImmType : WaveMachine_Type<"Imm", "imm"> {
  let summary = "machine immediate operand";
  let description = [{
    A `!wavemachine.imm` value represents an immediate machine operand. It keeps
    immediates in SSA form so generic WaveMachine operations can remain fully
    inspectable in textual MLIR.
  }];
}

def WaveMachine_MemTokenType : WaveMachine_Type<"MemToken", "mem.token"> {
  let summary = "machine memory dependency token";
  let description = [{
    A `!wavemachine.mem.token` value carries explicit WaveMachine memory
    dependencies for waitcnt insertion.
  }];
}

def WaveMachine_Reg
    : Type<CPred<"::llvm::isa<::mlir::wavemachine::RegType>($_self)">,
           "WaveMachine register type">;

def WaveMachine_Imm
    : Type<CPred<"::llvm::isa<::mlir::wavemachine::ImmType>($_self)">,
           "WaveMachine immediate type">;

def WaveMachine_MemToken
    : Type<CPred<"::llvm::isa<::mlir::wavemachine::MemTokenType>($_self)">,
           "WaveMachine memory token type">;

#endif // MLIR_DIALECT_WAVEMACHINE_IR_WAVEMACHINETYPES_TD

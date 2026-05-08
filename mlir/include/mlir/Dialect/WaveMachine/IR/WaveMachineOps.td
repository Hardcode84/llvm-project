//===-- WaveMachineOps.td - WaveMachine dialect ------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_WAVEMACHINE_IR_WAVEMACHINEOPS_TD
#define MLIR_DIALECT_WAVEMACHINE_IR_WAVEMACHINEOPS_TD

include "mlir/Dialect/WaveMachine/IR/WaveMachineBase.td"
include "mlir/Dialect/WaveMachine/IR/WaveMachineTypes.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

class WaveMachine_Op<string mnemonic, list<Trait> traits = []> :
    Op<WaveMachine_Dialect, mnemonic, traits>;

def WaveMachine_VALU : NativeOpTrait<"wavemachine::VALUOp">;
def WaveMachine_SMEMLoad : NativeOpTrait<"wavemachine::SMEMLoadOp">;
def WaveMachine_VMEMLoad : NativeOpTrait<"wavemachine::VMEMLoadOp">;
def WaveMachine_VMEMStore : NativeOpTrait<"wavemachine::VMEMStoreOp">;
def WaveMachine_Waitcnt : NativeOpTrait<"wavemachine::WaitcntOp">;
def WaveMachine_TokenTrait : NativeOpTrait<"wavemachine::TokenOp">;
def WaveMachine_TokenJoinTrait : NativeOpTrait<"wavemachine::TokenJoinOp">;

def WaveMachine_SGPR1
    : Type<CPred<"::llvm::isa<::mlir::wavemachine::RegType>($_self) && "
                 "::llvm::cast<::mlir::wavemachine::RegType>($_self).getRegClass() == ::mlir::wavemachine::RegClass::SGPR && "
                 "::llvm::cast<::mlir::wavemachine::RegType>($_self).getWidth() == 1">,
           "WaveMachine scalar SGPR register">;

def WaveMachine_SGPR2
    : Type<CPred<"::llvm::isa<::mlir::wavemachine::RegType>($_self) && "
                 "::llvm::cast<::mlir::wavemachine::RegType>($_self).getRegClass() == ::mlir::wavemachine::RegClass::SGPR && "
                 "::llvm::cast<::mlir::wavemachine::RegType>($_self).getWidth() == 2">,
           "WaveMachine SGPR pair register">;

def WaveMachine_SGPR4
    : Type<CPred<"::llvm::isa<::mlir::wavemachine::RegType>($_self) && "
                 "::llvm::cast<::mlir::wavemachine::RegType>($_self).getRegClass() == ::mlir::wavemachine::RegClass::SGPR && "
                 "::llvm::cast<::mlir::wavemachine::RegType>($_self).getWidth() == 4">,
           "WaveMachine SGPR quad register">;

def WaveMachine_VGPR
    : Type<CPred<"::llvm::isa<::mlir::wavemachine::RegType>($_self) && "
                 "::llvm::cast<::mlir::wavemachine::RegType>($_self).getRegClass() == ::mlir::wavemachine::RegClass::VGPR">,
           "WaveMachine VGPR register">;

def WaveMachine_VGPR1
    : Type<CPred<"::llvm::isa<::mlir::wavemachine::RegType>($_self) && "
                 "::llvm::cast<::mlir::wavemachine::RegType>($_self).getRegClass() == ::mlir::wavemachine::RegClass::VGPR && "
                 "::llvm::cast<::mlir::wavemachine::RegType>($_self).getWidth() == 1">,
           "WaveMachine scalar VGPR register">;

def WaveMachine_RegOrImm
    : Type<CPred<"::llvm::isa<::mlir::wavemachine::RegType>($_self) || "
                 "::llvm::isa<::mlir::wavemachine::ImmType>($_self)">,
           "WaveMachine register or immediate">;

class WaveMachine_ZeroResult<string mnemonic, list<Trait> traits = []>
    : WaveMachine_Op<mnemonic, traits> {
  let assemblyFormat = "attr-dict";
}

class WaveMachine_ZeroOperandResult<string mnemonic, Type resultType,
                                    list<Trait> traits = []>
    : WaveMachine_Op<mnemonic, traits> {
  let results = (outs resultType:$result);
  let assemblyFormat = "attr-dict `:` type($result)";
}

class WaveMachine_UnaryResult<string mnemonic, Type operandType, Type resultType,
                              list<Trait> traits = []>
    : WaveMachine_Op<mnemonic, traits> {
  let arguments = (ins operandType:$source);
  let results = (outs resultType:$result);
  let assemblyFormat = "$source attr-dict `:` functional-type(operands, results)";
}

class WaveMachine_BinaryResult<string mnemonic, Type operandType, Type resultType,
                               list<Trait> traits = []>
    : WaveMachine_Op<mnemonic, traits> {
  let arguments = (ins operandType:$lhs, operandType:$rhs);
  let results = (outs resultType:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` functional-type(operands, results)";
}

class WaveMachine_TernaryResult<string mnemonic, Type operandType, Type resultType,
                                list<Trait> traits = []>
    : WaveMachine_Op<mnemonic, traits> {
  let arguments = (ins operandType:$a, operandType:$b, operandType:$c);
  let results = (outs resultType:$result);
  let assemblyFormat = "$a `,` $b `,` $c attr-dict `:` functional-type(operands, results)";
}

class WaveMachine_UnaryNoResult<string mnemonic, Type operandType,
                                list<Trait> traits = []>
    : WaveMachine_Op<mnemonic, traits> {
  let arguments = (ins operandType:$source);
  let assemblyFormat = "$source attr-dict `:` functional-type(operands, results)";
}

class WaveMachine_BinaryNoResult<string mnemonic, Type operandType,
                                 list<Trait> traits = []>
    : WaveMachine_Op<mnemonic, traits> {
  let arguments = (ins operandType:$lhs, operandType:$rhs);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` functional-type(operands, results)";
}

def WaveMachine_ArgOp : WaveMachine_Op<"arg"> {
  let summary = "selected function argument";
  let arguments = (ins I64Attr:$index, BoolAttr:$pointer);
  let results = (outs WaveMachine_Reg:$result);
  let assemblyFormat = "attr-dict `:` type($result)";
}

def WaveMachine_ImmOp : WaveMachine_Op<"imm", [Pure]> {
  let summary = "machine immediate";
  let arguments = (ins I64Attr:$value);
  let results = (outs WaveMachine_Imm:$result);
  let assemblyFormat = "$value attr-dict `:` type($result)";
}

def WaveMachine_TokenOp : WaveMachine_Op<"token", [Pure, WaveMachine_TokenTrait]> {
  let summary = "empty machine memory token";
  let results = (outs WaveMachine_MemToken:$result);
  let assemblyFormat = "attr-dict `:` type($result)";
}

def WaveMachine_TokenJoinOp : WaveMachine_Op<"token_join",
    [Pure, WaveMachine_TokenJoinTrait]> {
  let summary = "join machine memory tokens";
  let arguments = (ins Variadic<WaveMachine_MemToken>:$dependencies);
  let results = (outs WaveMachine_MemToken:$result);
  let assemblyFormat = "$dependencies attr-dict `:` functional-type(operands, results)";
}

def WaveMachine_AfterOp : WaveMachine_Op<"after",
    [Pure, WaveMachine_TokenJoinTrait]> {
  let summary = "derive a machine memory token after dependencies";
  let arguments = (ins Variadic<WaveMachine_MemToken>:$dependencies);
  let results = (outs WaveMachine_MemToken:$result);
  let assemblyFormat = "$dependencies attr-dict `:` functional-type(operands, results)";
}

def WaveMachine_WaitOp : WaveMachine_Op<"wait"> {
  let summary = "wait for machine memory token dependencies";
  let arguments = (ins Variadic<WaveMachine_MemToken>:$dependencies);
  let assemblyFormat = "$dependencies attr-dict `:` functional-type(operands, results)";
}

def WaveMachine_LabelOp : WaveMachine_Op<"label"> {
  let summary = "assembly label";
  let arguments = (ins StrAttr:$name);
  let assemblyFormat = "$name attr-dict";
}

def WaveMachine_VMbcntLoOp
    : WaveMachine_ZeroOperandResult<"v_mbcnt_lo", WaveMachine_VGPR1,
                                    [WaveMachine_VALU]>;
def WaveMachine_VMovB32TupleOp
    : WaveMachine_UnaryResult<"v_mov_b32_tuple", WaveMachine_RegOrImm,
                              WaveMachine_VGPR, [WaveMachine_VALU]> {
  let hasVerifier = 1;
}
def WaveMachine_VAddU32Op
    : WaveMachine_BinaryResult<"v_add_u32", WaveMachine_RegOrImm,
                               WaveMachine_VGPR1, [WaveMachine_VALU]>;
def WaveMachine_VAndB32Op
    : WaveMachine_BinaryResult<"v_and_b32", WaveMachine_RegOrImm,
                               WaveMachine_VGPR1, [WaveMachine_VALU]>;
def WaveMachine_VOrB32Op
    : WaveMachine_BinaryResult<"v_or_b32", WaveMachine_RegOrImm,
                               WaveMachine_VGPR1, [WaveMachine_VALU]>;
def WaveMachine_VXorB32Op
    : WaveMachine_BinaryResult<"v_xor_b32", WaveMachine_RegOrImm,
                               WaveMachine_VGPR1, [WaveMachine_VALU]>;
def WaveMachine_VLshlrevB32Op
    : WaveMachine_BinaryResult<"v_lshlrev_b32", WaveMachine_RegOrImm,
                               WaveMachine_VGPR1, [WaveMachine_VALU]>;
def WaveMachine_VCmpEqU32Op
    : WaveMachine_BinaryResult<"v_cmp_eq_u32", WaveMachine_RegOrImm,
                               WaveMachine_SGPR1, [WaveMachine_VALU]>;
def WaveMachine_VCmpNeU32Op
    : WaveMachine_BinaryResult<"v_cmp_ne_u32", WaveMachine_RegOrImm,
                               WaveMachine_SGPR1, [WaveMachine_VALU]>;
def WaveMachine_VCmpLtU32Op
    : WaveMachine_BinaryResult<"v_cmp_lt_u32", WaveMachine_RegOrImm,
                               WaveMachine_SGPR1, [WaveMachine_VALU]>;
def WaveMachine_VCmpLeU32Op
    : WaveMachine_BinaryResult<"v_cmp_le_u32", WaveMachine_RegOrImm,
                               WaveMachine_SGPR1, [WaveMachine_VALU]>;
def WaveMachine_VCmpGtU32Op
    : WaveMachine_BinaryResult<"v_cmp_gt_u32", WaveMachine_RegOrImm,
                               WaveMachine_SGPR1, [WaveMachine_VALU]>;
def WaveMachine_VCmpGeU32Op
    : WaveMachine_BinaryResult<"v_cmp_ge_u32", WaveMachine_RegOrImm,
                               WaveMachine_SGPR1, [WaveMachine_VALU]>;
def WaveMachine_VReadfirstlaneB32Op
    : WaveMachine_UnaryResult<"v_readfirstlane_b32", WaveMachine_VGPR1,
                              WaveMachine_SGPR1, [WaveMachine_VALU]>;

def WaveMachine_WmmaI32_16x16x16_IU8Op
    : WaveMachine_TernaryResult<"wmma_i32_16x16x16_iu8", WaveMachine_VGPR,
                                WaveMachine_VGPR, [WaveMachine_VALU]> {
  let hasVerifier = 1;
}
def WaveMachine_WmmaF32_16x16x16_F16Op
    : WaveMachine_TernaryResult<"wmma_f32_16x16x16_f16", WaveMachine_VGPR,
                                WaveMachine_VGPR, [WaveMachine_VALU]> {
  let hasVerifier = 1;
}

def WaveMachine_SMovB32Op : WaveMachine_UnaryNoResult<"s_mov_b32",
                                                       WaveMachine_RegOrImm> {
  let arguments = (ins StrAttr:$dst, WaveMachine_RegOrImm:$source);
  let assemblyFormat = "$dst `,` $source attr-dict `:` functional-type(operands, results)";
}
def WaveMachine_SLoadB32Op : WaveMachine_UnaryResult<"s_load_b32",
                                                      WaveMachine_Imm,
                                                      WaveMachine_SGPR1,
                                                      [WaveMachine_SMEMLoad]> {
  let arguments = (ins WaveMachine_Imm:$offset, StrAttr:$base);
  let assemblyFormat = "$offset `,` $base attr-dict `:` functional-type(operands, results)";
}
def WaveMachine_SLoadB64Op : WaveMachine_UnaryResult<"s_load_b64",
                                                      WaveMachine_Imm,
                                                      WaveMachine_SGPR2,
                                                      [WaveMachine_SMEMLoad]> {
  let arguments = (ins WaveMachine_Imm:$offset, StrAttr:$base);
  let assemblyFormat = "$offset `,` $base attr-dict `:` functional-type(operands, results)";
}
def WaveMachine_SLoadB128Op : WaveMachine_UnaryResult<"s_load_b128",
                                                       WaveMachine_Imm,
                                                       WaveMachine_SGPR4,
                                                       [WaveMachine_SMEMLoad]> {
  let arguments = (ins WaveMachine_Imm:$offset, StrAttr:$base);
  let assemblyFormat = "$offset `,` $base attr-dict `:` functional-type(operands, results)";
}
def WaveMachine_SWaitcntOp : WaveMachine_UnaryNoResult<"s_waitcnt",
                                                        WaveMachine_Imm,
                                                        [WaveMachine_Waitcnt]>;
def WaveMachine_SWaitcntVscntOp : WaveMachine_UnaryNoResult<"s_waitcnt_vscnt",
                                                             WaveMachine_Imm,
                                                             [WaveMachine_Waitcnt]>;
def WaveMachine_SNopOp : WaveMachine_UnaryNoResult<"s_nop", WaveMachine_Imm>;
def WaveMachine_SDelayAluOp : WaveMachine_UnaryNoResult<"s_delay_alu",
                                                         WaveMachine_Imm>;
def WaveMachine_SAndSaveexecB32Op
    : WaveMachine_UnaryResult<"s_and_saveexec_b32", WaveMachine_SGPR1,
                              WaveMachine_SGPR1>;
def WaveMachine_SAndn2ExecB32Op
    : WaveMachine_BinaryNoResult<"s_andn2_exec_b32", WaveMachine_SGPR1>;
def WaveMachine_SCBranchExeczOp : WaveMachine_ZeroResult<"s_cbranch_execz"> {
  let arguments = (ins StrAttr:$label);
  let assemblyFormat = "$label attr-dict";
}
def WaveMachine_SMovExecLoOp : WaveMachine_UnaryNoResult<"s_mov_exec_lo",
                                                          WaveMachine_SGPR1>;
def WaveMachine_SEndpgmOp : WaveMachine_ZeroResult<"s_endpgm">;
def WaveMachine_SSetpcB64Op : WaveMachine_ZeroResult<"s_setpc_b64">;

def WaveMachine_GlobalLoadB32Op
    : WaveMachine_Op<"global_load_b32", [WaveMachine_VMEMLoad]> {
  let summary = "global 32-bit load through an SGPR base";
  let arguments = (ins WaveMachine_VGPR1:$offset, WaveMachine_SGPR2:$base,
                       Optional<WaveMachine_MemToken>:$dependency);
  let results = (outs WaveMachine_VGPR1:$result);
  let assemblyFormat = "$offset `,` $base (`after` $dependency^)? attr-dict `:` functional-type(operands, results)";
}

def WaveMachine_MakeBufferRsrcOp
    : WaveMachine_Op<"make_buffer_rsrc", [Pure]> {
  let summary = "construct an AMD buffer descriptor";
  let arguments = (ins WaveMachine_SGPR2:$base,
                       WaveMachine_RegOrImm:$range);
  let results = (outs WaveMachine_SGPR4:$descriptor);
  let assemblyFormat = "$base `,` $range attr-dict `:` functional-type(operands, results)";
}

def WaveMachine_GlobalStoreB32Op
    : WaveMachine_Op<"global_store_b32", [WaveMachine_VMEMStore]> {
  let summary = "global 32-bit store through an SGPR base";
  let arguments = (ins WaveMachine_VGPR1:$offset, WaveMachine_RegOrImm:$value,
                       WaveMachine_SGPR2:$base,
                       Optional<WaveMachine_MemToken>:$dependency);
  let results = (outs Variadic<WaveMachine_MemToken>:$tokens);
  let assemblyFormat = "$offset `,` $value `,` $base (`after` $dependency^)? attr-dict `:` functional-type(operands, results)";
  let hasVerifier = 1;
}
def WaveMachine_BufferStoreB32Op
    : WaveMachine_Op<"buffer_store_b32", [WaveMachine_VMEMStore]> {
  let summary = "buffer 32-bit store through an SGPR descriptor and VGPR offset";
  let arguments = (ins WaveMachine_VGPR1:$offset, WaveMachine_VGPR1:$value,
                       WaveMachine_SGPR4:$descriptor,
                       Optional<WaveMachine_MemToken>:$dependency);
  let results = (outs Variadic<WaveMachine_MemToken>:$tokens);
  let assemblyFormat = "$offset `,` $value `,` $descriptor (`after` $dependency^)? attr-dict `:` functional-type(operands, results)";
  let hasVerifier = 1;
}
def WaveMachine_GlobalStoreTupleB32Op
    : WaveMachine_Op<"global_store_tuple_b32", [WaveMachine_VMEMStore]> {
  let summary = "global 32-bit store of a VGPR tuple component";
  let arguments = (ins WaveMachine_VGPR1:$offset, WaveMachine_VGPR:$value,
                       WaveMachine_SGPR2:$base,
                       Optional<WaveMachine_MemToken>:$dependency);
  let results = (outs Variadic<WaveMachine_MemToken>:$tokens);
  let assemblyFormat = "$offset `,` $value `,` $base (`after` $dependency^)? attr-dict `:` functional-type(operands, results)";
  let hasVerifier = 1;
}

#endif // MLIR_DIALECT_WAVEMACHINE_IR_WAVEMACHINEOPS_TD

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

class WaveMachine_GenericOp<string mnemonic, list<Trait> traits = []>
    : WaveMachine_Op<mnemonic, traits> {
  let arguments = (ins Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>:$results);
}

def WaveMachine_ArgOp : WaveMachine_Op<"arg"> {
  let summary = "selected function argument";
  let results = (outs WaveMachine_Reg:$result);
}

def WaveMachine_ImmOp : WaveMachine_Op<"imm", [Pure]> {
  let summary = "machine immediate";
  let results = (outs WaveMachine_Imm:$result);
}

def WaveMachine_TokenOp : WaveMachine_Op<"token", [Pure, WaveMachine_TokenTrait]> {
  let summary = "empty machine memory token";
  let results = (outs WaveMachine_MemToken:$result);
}

def WaveMachine_TokenJoinOp : WaveMachine_Op<"token_join",
    [Pure, WaveMachine_TokenJoinTrait]> {
  let summary = "join machine memory tokens";
  let arguments = (ins Variadic<WaveMachine_MemToken>:$dependencies);
  let results = (outs WaveMachine_MemToken:$result);
}

def WaveMachine_AfterOp : WaveMachine_Op<"after",
    [Pure, WaveMachine_TokenJoinTrait]> {
  let summary = "derive a machine memory token after dependencies";
  let arguments = (ins Variadic<WaveMachine_MemToken>:$dependencies);
  let results = (outs WaveMachine_MemToken:$result);
}

def WaveMachine_WaitOp : WaveMachine_Op<"wait"> {
  let summary = "wait for machine memory token dependencies";
  let arguments = (ins Variadic<WaveMachine_MemToken>:$dependencies);
}

def WaveMachine_LabelOp : WaveMachine_Op<"label">;

def WaveMachine_VMbcntLoOp : WaveMachine_GenericOp<"v_mbcnt_lo", [WaveMachine_VALU]>;
def WaveMachine_VMovB32TupleOp : WaveMachine_GenericOp<"v_mov_b32_tuple", [WaveMachine_VALU]>;
def WaveMachine_VAddU32Op : WaveMachine_GenericOp<"v_add_u32", [WaveMachine_VALU]>;
def WaveMachine_VAndB32Op : WaveMachine_GenericOp<"v_and_b32", [WaveMachine_VALU]>;
def WaveMachine_VOrB32Op : WaveMachine_GenericOp<"v_or_b32", [WaveMachine_VALU]>;
def WaveMachine_VXorB32Op : WaveMachine_GenericOp<"v_xor_b32", [WaveMachine_VALU]>;
def WaveMachine_VLshlrevB32Op : WaveMachine_GenericOp<"v_lshlrev_b32", [WaveMachine_VALU]>;
def WaveMachine_VCmpEqU32Op : WaveMachine_GenericOp<"v_cmp_eq_u32", [WaveMachine_VALU]>;
def WaveMachine_VCmpNeU32Op : WaveMachine_GenericOp<"v_cmp_ne_u32", [WaveMachine_VALU]>;
def WaveMachine_VCmpLtU32Op : WaveMachine_GenericOp<"v_cmp_lt_u32", [WaveMachine_VALU]>;
def WaveMachine_VCmpLeU32Op : WaveMachine_GenericOp<"v_cmp_le_u32", [WaveMachine_VALU]>;
def WaveMachine_VCmpGtU32Op : WaveMachine_GenericOp<"v_cmp_gt_u32", [WaveMachine_VALU]>;
def WaveMachine_VCmpGeU32Op : WaveMachine_GenericOp<"v_cmp_ge_u32", [WaveMachine_VALU]>;
def WaveMachine_VReadfirstlaneB32Op : WaveMachine_GenericOp<"v_readfirstlane_b32", [WaveMachine_VALU]>;

def WaveMachine_WmmaI32_16x16x16_IU8Op
    : WaveMachine_GenericOp<"wmma_i32_16x16x16_iu8", [WaveMachine_VALU]>;
def WaveMachine_WmmaF32_16x16x16_F16Op
    : WaveMachine_GenericOp<"wmma_f32_16x16x16_f16", [WaveMachine_VALU]>;

def WaveMachine_SMovB32Op : WaveMachine_GenericOp<"s_mov_b32">;
def WaveMachine_SLoadB32Op : WaveMachine_GenericOp<"s_load_b32", [WaveMachine_SMEMLoad]>;
def WaveMachine_SLoadB64Op : WaveMachine_GenericOp<"s_load_b64", [WaveMachine_SMEMLoad]>;
def WaveMachine_SWaitcntOp : WaveMachine_GenericOp<"s_waitcnt", [WaveMachine_Waitcnt]>;
def WaveMachine_SWaitcntVscntOp : WaveMachine_GenericOp<"s_waitcnt_vscnt", [WaveMachine_Waitcnt]>;
def WaveMachine_SDelayAluOp : WaveMachine_GenericOp<"s_delay_alu">;
def WaveMachine_SAndSaveexecB32Op : WaveMachine_GenericOp<"s_and_saveexec_b32">;
def WaveMachine_SAndn2ExecB32Op : WaveMachine_GenericOp<"s_andn2_exec_b32">;
def WaveMachine_SCBranchExeczOp : WaveMachine_GenericOp<"s_cbranch_execz">;
def WaveMachine_SMovExecLoOp : WaveMachine_GenericOp<"s_mov_exec_lo">;
def WaveMachine_SEndpgmOp : WaveMachine_GenericOp<"s_endpgm">;
def WaveMachine_SSetpcB64Op : WaveMachine_GenericOp<"s_setpc_b64">;

def WaveMachine_GlobalLoadB32Op : WaveMachine_GenericOp<"global_load_b32", [WaveMachine_VMEMLoad]>;
def WaveMachine_GlobalStoreB32Op : WaveMachine_GenericOp<"global_store_b32", [WaveMachine_VMEMStore]>;
def WaveMachine_GlobalStoreTupleB32Op
    : WaveMachine_GenericOp<"global_store_tuple_b32", [WaveMachine_VMEMStore]>;

#endif // MLIR_DIALECT_WAVEMACHINE_IR_WAVEMACHINEOPS_TD

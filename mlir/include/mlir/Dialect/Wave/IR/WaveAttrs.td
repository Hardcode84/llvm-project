//===-- WaveAttrs.td - Wave dialect attributes ------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_WAVE_IR_WAVEATTRS_TD
#define MLIR_DIALECT_WAVE_IR_WAVEATTRS_TD

include "mlir/Dialect/Wave/IR/WaveBase.td"
include "mlir/IR/AttrTypeBase.td"

class Wave_Attr<string name, string attrMnemonic, list<Trait> traits = []>
    : AttrDef<Wave_Dialect, name, traits> {
  let mnemonic = attrMnemonic;
}

def Wave_GlobalAddressSpaceAttr : Wave_Attr<"GlobalAddressSpace", "global"> {
  let summary = "global memory address space";
  let assemblyFormat = "";
}

def Wave_SharedAddressSpaceAttr : Wave_Attr<"SharedAddressSpace", "shared"> {
  let summary = "shared/workgroup memory address space";
  let assemblyFormat = "";
}

def Wave_PrivateAddressSpaceAttr : Wave_Attr<"PrivateAddressSpace", "private"> {
  let summary = "private memory address space";
  let assemblyFormat = "";
}

#endif // MLIR_DIALECT_WAVE_IR_WAVEATTRS_TD

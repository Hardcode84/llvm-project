//===-- WaveAMDAttrs.td - WaveAMD dialect attributes --------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_WAVE_IR_WAVEAMDATTRS_TD
#define MLIR_DIALECT_WAVE_IR_WAVEAMDATTRS_TD

include "mlir/Dialect/Wave/IR/WaveAMDBase.td"
include "mlir/IR/AttrTypeBase.td"

class WaveAMD_Attr<string name, string attrMnemonic, list<Trait> traits = []>
    : AttrDef<WaveAMD_Dialect, name, traits> {
  let mnemonic = attrMnemonic;
}

def WaveAMD_BufferAddressSpaceAttr
    : WaveAMD_Attr<"BufferAddressSpace", "buffer"> {
  let summary = "AMD buffer resource address space";
  let description = [{
    Marks a `!wave.ptr` as referring to an AMD buffer/resource-style address
    space. The target-independent `wave` pointer type accepts this as an
    arbitrary address-space attribute; target-specific lowering can later choose
    a descriptor/resource representation.
  }];
  let assemblyFormat = "";
}

#endif // MLIR_DIALECT_WAVE_IR_WAVEAMDATTRS_TD

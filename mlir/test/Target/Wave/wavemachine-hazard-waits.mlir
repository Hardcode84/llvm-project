// RUN: mlir-opt --waveamd-insert-hazard-waits -split-input-file %s | FileCheck %s

module attributes {wavemachine.target = "amdgcn-amd-amdhsa--gfx1100"} {

// CHECK-LABEL: func.func @delay_after_lgkm_wait
// CHECK: wavemachine.s_waitcnt
// CHECK-NEXT: wavemachine.imm 1
// CHECK-NEXT: wavemachine.s_delay_alu
// CHECK-NEXT: wavemachine.v_add_u32
func.func @delay_after_lgkm_wait(%x: !wavemachine.reg<vgpr, 1>, %y: !wavemachine.reg<sgpr, 1>) {
  %wait = wavemachine.imm 64519 : !wavemachine.imm
  wavemachine.s_waitcnt %wait : (!wavemachine.imm) -> ()
  %sum = wavemachine.v_add_u32 %x, %y : (!wavemachine.reg<vgpr, 1>, !wavemachine.reg<sgpr, 1>) -> !wavemachine.reg<vgpr, 1>
  return
}

// CHECK-LABEL: func.func @no_delay_after_vmcnt_wait
// CHECK: wavemachine.s_waitcnt
// CHECK-NEXT: wavemachine.v_add_u32
func.func @no_delay_after_vmcnt_wait(%x: !wavemachine.reg<vgpr, 1>, %y: !wavemachine.reg<sgpr, 1>) {
  %wait = wavemachine.imm 1023 : !wavemachine.imm
  wavemachine.s_waitcnt %wait : (!wavemachine.imm) -> ()
  %sum = wavemachine.v_add_u32 %x, %y : (!wavemachine.reg<vgpr, 1>, !wavemachine.reg<sgpr, 1>) -> !wavemachine.reg<vgpr, 1>
  return
}

}

// -----

module attributes {wavemachine.target = "amdgcn-amd-amdhsa--gfx1030"} {

// CHECK-LABEL: func.func @nop_delay_on_gfx10
// CHECK: wavemachine.s_waitcnt
// CHECK-NEXT: wavemachine.imm 0
// CHECK-NEXT: wavemachine.s_nop
// CHECK-NEXT: wavemachine.v_add_u32
func.func @nop_delay_on_gfx10(%x: !wavemachine.reg<vgpr, 1>, %y: !wavemachine.reg<sgpr, 1>) {
  %wait = wavemachine.imm 0 : !wavemachine.imm
  wavemachine.s_waitcnt %wait : (!wavemachine.imm) -> ()
  %sum = wavemachine.v_add_u32 %x, %y : (!wavemachine.reg<vgpr, 1>, !wavemachine.reg<sgpr, 1>) -> !wavemachine.reg<vgpr, 1>
  return
}

}

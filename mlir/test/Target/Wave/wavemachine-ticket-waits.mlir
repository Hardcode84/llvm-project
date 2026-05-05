// RUN: mlir-opt --waveamd-insert-ticket-waits -split-input-file %s | FileCheck %s

module attributes {wavemachine.target = "amdgcn-amd-amdhsa--gfx1100"} {

// CHECK-LABEL: func.func @lgkm_nonzero_distance
// CHECK: wavemachine.s_load_b32
// CHECK: wavemachine.s_load_b32
// CHECK: wavemachine.imm 64535
// CHECK-NEXT: wavemachine.s_waitcnt
// CHECK-NEXT: wavemachine.v_add_u32
func.func @lgkm_nonzero_distance(%x: !wavemachine.reg<vgpr, 1>) {
  %zero = wavemachine.imm 0 : !wavemachine.imm
  %a = wavemachine.s_load_b32 %zero, "s[0:1]" : (!wavemachine.imm) -> !wavemachine.reg<sgpr, 1>
  %b = wavemachine.s_load_b32 %zero, "s[0:1]" : (!wavemachine.imm) -> !wavemachine.reg<sgpr, 1>
  %sum = wavemachine.v_add_u32 %x, %a : (!wavemachine.reg<vgpr, 1>, !wavemachine.reg<sgpr, 1>) -> !wavemachine.reg<vgpr, 1>
  return
}

}

// -----

module attributes {wavemachine.target = "amdgcn-amd-amdhsa--gfx1100"} {

// CHECK-LABEL: func.func @store_uses_vscnt
// CHECK: wavemachine.global_store_b32
// CHECK: wavemachine.imm 0
// CHECK-NEXT: wavemachine.s_waitcnt_vscnt
// CHECK-NEXT: wavemachine.s_endpgm
func.func @store_uses_vscnt(%offset: !wavemachine.reg<vgpr, 1>, %value: !wavemachine.reg<vgpr, 1>, %base: !wavemachine.reg<sgpr, 2>) {
  wavemachine.global_store_b32 %offset, %value, %base : (!wavemachine.reg<vgpr, 1>, !wavemachine.reg<vgpr, 1>, !wavemachine.reg<sgpr, 2>) -> ()
  wavemachine.s_endpgm
  return
}

}

// -----

module attributes {wavemachine.target = "amdgcn-amd-amdhsa--gfx1100"} {

// CHECK-LABEL: func.func @existing_wait_satisfies_use
// CHECK: wavemachine.s_waitcnt
// CHECK-NOT: wavemachine.s_waitcnt
// CHECK: wavemachine.v_add_u32
func.func @existing_wait_satisfies_use(%x: !wavemachine.reg<vgpr, 1>) {
  %zero = wavemachine.imm 0 : !wavemachine.imm
  %a = wavemachine.s_load_b32 %zero, "s[0:1]" : (!wavemachine.imm) -> !wavemachine.reg<sgpr, 1>
  %wait = wavemachine.imm 64519 : !wavemachine.imm
  wavemachine.s_waitcnt %wait : (!wavemachine.imm) -> ()
  %sum = wavemachine.v_add_u32 %x, %a : (!wavemachine.reg<vgpr, 1>, !wavemachine.reg<sgpr, 1>) -> !wavemachine.reg<vgpr, 1>
  return
}

}

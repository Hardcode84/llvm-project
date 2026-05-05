// RUN: mlir-opt --wavemachine-insert-hazard-waits %s | FileCheck %s

module attributes {wavemachine.target = "amdgcn-amd-amdhsa--gfx1100"} {

// CHECK-LABEL: func.func @delay_after_lgkm_wait
// CHECK: "wavemachine.s_waitcnt"
// CHECK-NEXT: "wavemachine.imm"() {value = 1 : i64}
// CHECK-NEXT: "wavemachine.s_delay_alu"
// CHECK-NEXT: "wavemachine.v_add_u32"
func.func @delay_after_lgkm_wait(%x: !wavemachine.reg<1, 1>, %y: !wavemachine.reg<0, 1>) {
  %wait = "wavemachine.imm"() {value = 64519 : i64} : () -> !wavemachine.imm
  "wavemachine.s_waitcnt"(%wait) : (!wavemachine.imm) -> ()
  %sum = "wavemachine.v_add_u32"(%x, %y) : (!wavemachine.reg<1, 1>, !wavemachine.reg<0, 1>) -> !wavemachine.reg<1, 1>
  return
}

// CHECK-LABEL: func.func @no_delay_after_vmcnt_wait
// CHECK: "wavemachine.s_waitcnt"
// CHECK-NEXT: "wavemachine.v_add_u32"
func.func @no_delay_after_vmcnt_wait(%x: !wavemachine.reg<1, 1>, %y: !wavemachine.reg<0, 1>) {
  %wait = "wavemachine.imm"() {value = 1023 : i64} : () -> !wavemachine.imm
  "wavemachine.s_waitcnt"(%wait) : (!wavemachine.imm) -> ()
  %sum = "wavemachine.v_add_u32"(%x, %y) : (!wavemachine.reg<1, 1>, !wavemachine.reg<0, 1>) -> !wavemachine.reg<1, 1>
  return
}

}

// RUN: mlir-opt --wavemachine-insert-ticket-waits -split-input-file %s | FileCheck %s

// CHECK-LABEL: func.func @lgkm_nonzero_distance
// CHECK: "wavemachine.s_load_b32"
// CHECK: "wavemachine.s_load_b32"
// CHECK: "wavemachine.imm"() {value = 64535 : i64}
// CHECK-NEXT: "wavemachine.s_waitcnt"
// CHECK-NEXT: "wavemachine.v_add_u32"
func.func @lgkm_nonzero_distance(%x: !wavemachine.reg<1, 1>) {
  %zero = "wavemachine.imm"() {value = 0 : i64} : () -> !wavemachine.imm
  %a = "wavemachine.s_load_b32"(%zero) {base = "s[0:1]"} : (!wavemachine.imm) -> !wavemachine.reg<0, 1>
  %b = "wavemachine.s_load_b32"(%zero) {base = "s[0:1]"} : (!wavemachine.imm) -> !wavemachine.reg<0, 1>
  %sum = "wavemachine.v_add_u32"(%x, %a) : (!wavemachine.reg<1, 1>, !wavemachine.reg<0, 1>) -> !wavemachine.reg<1, 1>
  return
}

// -----

// CHECK-LABEL: func.func @store_uses_vscnt
// CHECK: "wavemachine.global_store_b32"
// CHECK: "wavemachine.imm"() {value = 0 : i64}
// CHECK-NEXT: "wavemachine.s_waitcnt_vscnt"
// CHECK-NEXT: "wavemachine.s_endpgm"
func.func @store_uses_vscnt(%offset: !wavemachine.reg<1, 1>, %value: !wavemachine.reg<1, 1>, %base: !wavemachine.reg<0, 2>) {
  "wavemachine.global_store_b32"(%offset, %value, %base) : (!wavemachine.reg<1, 1>, !wavemachine.reg<1, 1>, !wavemachine.reg<0, 2>) -> ()
  "wavemachine.s_endpgm"() : () -> ()
  return
}

// -----

// CHECK-LABEL: func.func @existing_wait_satisfies_use
// CHECK: "wavemachine.s_waitcnt"
// CHECK-NOT: "wavemachine.s_waitcnt"
// CHECK: "wavemachine.v_add_u32"
func.func @existing_wait_satisfies_use(%x: !wavemachine.reg<1, 1>) {
  %zero = "wavemachine.imm"() {value = 0 : i64} : () -> !wavemachine.imm
  %a = "wavemachine.s_load_b32"(%zero) {base = "s[0:1]"} : (!wavemachine.imm) -> !wavemachine.reg<0, 1>
  %wait = "wavemachine.imm"() {value = 64519 : i64} : () -> !wavemachine.imm
  "wavemachine.s_waitcnt"(%wait) : (!wavemachine.imm) -> ()
  %sum = "wavemachine.v_add_u32"(%x, %a) : (!wavemachine.reg<1, 1>, !wavemachine.reg<0, 1>) -> !wavemachine.reg<1, 1>
  return
}

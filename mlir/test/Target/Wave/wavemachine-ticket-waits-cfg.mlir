// RUN: mlir-opt --wavemachine-insert-ticket-waits -split-input-file %s | FileCheck %s

// CHECK-LABEL: func.func @cfg_join_nonzero
// CHECK: "wavemachine.s_load_b32"
// CHECK: cf.cond_br
// CHECK: "wavemachine.s_load_b32"
// CHECK: cf.br
// CHECK: ^bb{{[0-9]+}}(%{{[0-9]+}}: !wavemachine.reg<0, 1>)
// CHECK-NEXT: "wavemachine.imm"() {value = 64535 : i64}
// CHECK-NEXT: "wavemachine.s_waitcnt"
// CHECK-NEXT: "wavemachine.imm"() {value = 1 : i64}
// CHECK-NEXT: "wavemachine.s_delay_alu"
// CHECK-NEXT: "wavemachine.v_add_u32"
func.func @cfg_join_nonzero(%cond: i1, %x: !wavemachine.reg<1, 1>) {
  %zero = "wavemachine.imm"() {value = 0 : i64} : () -> !wavemachine.imm
  %a = "wavemachine.s_load_b32"(%zero) {base = "s[0:1]"} : (!wavemachine.imm) -> !wavemachine.reg<0, 1>
  cf.cond_br %cond, ^then, ^else
^then:
  %b = "wavemachine.s_load_b32"(%zero) {base = "s[0:1]"} : (!wavemachine.imm) -> !wavemachine.reg<0, 1>
  cf.br ^merge(%a : !wavemachine.reg<0, 1>)
^else:
  cf.br ^merge(%a : !wavemachine.reg<0, 1>)
^merge(%m: !wavemachine.reg<0, 1>):
  %sum = "wavemachine.v_add_u32"(%x, %m) : (!wavemachine.reg<1, 1>, !wavemachine.reg<0, 1>) -> !wavemachine.reg<1, 1>
  return
}

// -----

// CHECK-LABEL: func.func @block_arg_ticket
// CHECK: "wavemachine.s_load_b32"
// CHECK: cf.br
// CHECK: ^bb{{[0-9]+}}(%{{[0-9]+}}: !wavemachine.reg<0, 1>)
// CHECK-NEXT: "wavemachine.imm"() {value = 64519 : i64}
// CHECK-NEXT: "wavemachine.s_waitcnt"
// CHECK-NEXT: "wavemachine.imm"() {value = 1 : i64}
// CHECK-NEXT: "wavemachine.s_delay_alu"
// CHECK-NEXT: "wavemachine.v_add_u32"
func.func @block_arg_ticket(%x: !wavemachine.reg<1, 1>) {
  %zero = "wavemachine.imm"() {value = 0 : i64} : () -> !wavemachine.imm
  %a = "wavemachine.s_load_b32"(%zero) {base = "s[0:1]"} : (!wavemachine.imm) -> !wavemachine.reg<0, 1>
  cf.br ^merge(%a : !wavemachine.reg<0, 1>)
^merge(%m: !wavemachine.reg<0, 1>):
  %sum = "wavemachine.v_add_u32"(%x, %m) : (!wavemachine.reg<1, 1>, !wavemachine.reg<0, 1>) -> !wavemachine.reg<1, 1>
  return
}

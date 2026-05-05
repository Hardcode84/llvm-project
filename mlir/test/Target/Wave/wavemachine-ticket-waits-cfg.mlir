// RUN: mlir-opt --waveamd-insert-ticket-waits -split-input-file %s | FileCheck %s

module attributes {wavemachine.target = "amdgcn-amd-amdhsa--gfx1100"} {

// CHECK-LABEL: func.func @cfg_join_nonzero
// CHECK: wavemachine.s_load_b32
// CHECK: cf.cond_br
// CHECK: wavemachine.s_load_b32
// CHECK: cf.br
// CHECK: ^bb{{[0-9]+}}(%{{[0-9]+}}: !wavemachine.reg<sgpr, 1>)
// CHECK-NEXT: wavemachine.imm 64535
// CHECK-NEXT: wavemachine.s_waitcnt
// CHECK-NEXT: wavemachine.v_add_u32
func.func @cfg_join_nonzero(%cond: i1, %x: !wavemachine.reg<vgpr, 1>) {
  %zero = wavemachine.imm 0 : !wavemachine.imm
  %a = wavemachine.s_load_b32 %zero, "s[0:1]" : (!wavemachine.imm) -> !wavemachine.reg<sgpr, 1>
  cf.cond_br %cond, ^then, ^else
^then:
  %b = wavemachine.s_load_b32 %zero, "s[0:1]" : (!wavemachine.imm) -> !wavemachine.reg<sgpr, 1>
  cf.br ^merge(%a : !wavemachine.reg<sgpr, 1>)
^else:
  cf.br ^merge(%a : !wavemachine.reg<sgpr, 1>)
^merge(%m: !wavemachine.reg<sgpr, 1>):
  %sum = wavemachine.v_add_u32 %x, %m : (!wavemachine.reg<vgpr, 1>, !wavemachine.reg<sgpr, 1>) -> !wavemachine.reg<vgpr, 1>
  return
}

}

// -----

module attributes {wavemachine.target = "amdgcn-amd-amdhsa--gfx1100"} {

// CHECK-LABEL: func.func @block_arg_ticket
// CHECK: wavemachine.s_load_b32
// CHECK: cf.br
// CHECK: ^bb{{[0-9]+}}(%{{[0-9]+}}: !wavemachine.reg<sgpr, 1>)
// CHECK-NEXT: wavemachine.imm 64519
// CHECK-NEXT: wavemachine.s_waitcnt
// CHECK-NEXT: wavemachine.v_add_u32
func.func @block_arg_ticket(%x: !wavemachine.reg<vgpr, 1>) {
  %zero = wavemachine.imm 0 : !wavemachine.imm
  %a = wavemachine.s_load_b32 %zero, "s[0:1]" : (!wavemachine.imm) -> !wavemachine.reg<sgpr, 1>
  cf.br ^merge(%a : !wavemachine.reg<sgpr, 1>)
^merge(%m: !wavemachine.reg<sgpr, 1>):
  %sum = wavemachine.v_add_u32 %x, %m : (!wavemachine.reg<vgpr, 1>, !wavemachine.reg<sgpr, 1>) -> !wavemachine.reg<vgpr, 1>
  return
}

}

// -----

module attributes {wavemachine.target = "amdgcn-amd-amdhsa--gfx1100"} {

// CHECK-LABEL: func.func @structured_if_nonzero
// CHECK: scf.if
// CHECK: wavemachine.s_load_b32
// CHECK: wavemachine.imm 64535
// CHECK-NEXT: wavemachine.s_waitcnt
// CHECK-NEXT: wavemachine.v_add_u32
func.func @structured_if_nonzero(%cond: i1, %x: !wavemachine.reg<vgpr, 1>) {
  %zero = wavemachine.imm 0 : !wavemachine.imm
  %a = wavemachine.s_load_b32 %zero, "s[0:1]" : (!wavemachine.imm) -> !wavemachine.reg<sgpr, 1>
  %r = scf.if %cond -> (!wavemachine.reg<sgpr, 1>) {
    %b = wavemachine.s_load_b32 %zero, "s[0:1]" : (!wavemachine.imm) -> !wavemachine.reg<sgpr, 1>
    scf.yield %a : !wavemachine.reg<sgpr, 1>
  } else {
    scf.yield %a : !wavemachine.reg<sgpr, 1>
  }
  %sum = wavemachine.v_add_u32 %x, %r : (!wavemachine.reg<vgpr, 1>, !wavemachine.reg<sgpr, 1>) -> !wavemachine.reg<vgpr, 1>
  return
}

}

// -----

module attributes {wavemachine.target = "amdgcn-amd-amdhsa--gfx1100"} {

// CHECK-LABEL: func.func @structured_for_double_buffer
// CHECK: scf.for
// CHECK: wavemachine.s_load_b32
// CHECK-NEXT: wavemachine.imm 64535
// CHECK-NEXT: wavemachine.s_waitcnt
// CHECK-NEXT: wavemachine.v_add_u32
func.func @structured_for_double_buffer(%x: !wavemachine.reg<vgpr, 1>) {
  %zero = wavemachine.imm 0 : !wavemachine.imm
  %init = wavemachine.s_load_b32 %zero, "s[0:1]" : (!wavemachine.imm) -> !wavemachine.reg<sgpr, 1>
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %c1 = arith.constant 1 : index
  %res = scf.for %i = %c0 to %c4 step %c1 iter_args(%cur = %init) -> (!wavemachine.reg<sgpr, 1>) {
    %next = wavemachine.s_load_b32 %zero, "s[0:1]" : (!wavemachine.imm) -> !wavemachine.reg<sgpr, 1>
    %sum = wavemachine.v_add_u32 %x, %cur : (!wavemachine.reg<vgpr, 1>, !wavemachine.reg<sgpr, 1>) -> !wavemachine.reg<vgpr, 1>
    scf.yield %next : !wavemachine.reg<sgpr, 1>
  }
  return
}

}

// -----

module attributes {wavemachine.target = "amdgcn-amd-amdhsa--gfx1100"} {

// CHECK-LABEL: func.func @structured_for_triple_buffer
// CHECK: scf.for
// CHECK: wavemachine.s_load_b32
// CHECK-NEXT: wavemachine.imm 64551
// CHECK-NEXT: wavemachine.s_waitcnt
// CHECK-NEXT: wavemachine.v_add_u32
func.func @structured_for_triple_buffer(%x: !wavemachine.reg<vgpr, 1>) {
  %zero = wavemachine.imm 0 : !wavemachine.imm
  %init0 = wavemachine.s_load_b32 %zero, "s[0:1]" : (!wavemachine.imm) -> !wavemachine.reg<sgpr, 1>
  %init1 = wavemachine.s_load_b32 %zero, "s[0:1]" : (!wavemachine.imm) -> !wavemachine.reg<sgpr, 1>
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %c1 = arith.constant 1 : index
  %r0, %r1 = scf.for %i = %c0 to %c4 step %c1
      iter_args(%cur = %init0, %nextBuf = %init1)
      -> (!wavemachine.reg<sgpr, 1>, !wavemachine.reg<sgpr, 1>) {
    %future0 = wavemachine.s_load_b32 %zero, "s[0:1]" : (!wavemachine.imm) -> !wavemachine.reg<sgpr, 1>
    %sum = wavemachine.v_add_u32 %x, %cur : (!wavemachine.reg<vgpr, 1>, !wavemachine.reg<sgpr, 1>) -> !wavemachine.reg<vgpr, 1>
    scf.yield %nextBuf, %future0 : !wavemachine.reg<sgpr, 1>, !wavemachine.reg<sgpr, 1>
  }
  return
}

}

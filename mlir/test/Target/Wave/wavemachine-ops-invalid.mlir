// RUN: mlir-opt -split-input-file -verify-diagnostics %s

func.func @bad_v_add_operand_count(%x: !wavemachine.reg<vgpr, 1>) {
  // expected-error @below {{'wavemachine.v_add_u32' op expected 2 operands}}
  %r = "wavemachine.v_add_u32"(%x) : (!wavemachine.reg<vgpr, 1>) -> !wavemachine.reg<vgpr, 1>
  return
}

// -----

func.func @bad_v_mbcnt_result_type() {
  // expected-error @below {{result #0 must be WaveMachine scalar VGPR register}}
  %r = "wavemachine.v_mbcnt_lo"() : () -> !wavemachine.reg<sgpr, 1>
  return
}

// -----

func.func @bad_s_load_result_type(%offset: !wavemachine.imm) {
  // expected-error @below {{result #0 must be WaveMachine SGPR pair register}}
  %r = "wavemachine.s_load_b64"(%offset) {base = "s[0:1]"} : (!wavemachine.imm) -> !wavemachine.reg<sgpr, 1>
  return
}

// -----

func.func @bad_global_store_base(%offset: !wavemachine.reg<vgpr, 1>, %value: !wavemachine.reg<vgpr, 1>, %base: !wavemachine.reg<sgpr, 1>) {
  // expected-error @below {{operand #2 must be WaveMachine SGPR pair register}}
  "wavemachine.global_store_b32"(%offset, %value, %base) : (!wavemachine.reg<vgpr, 1>, !wavemachine.reg<vgpr, 1>, !wavemachine.reg<sgpr, 1>) -> ()
  return
}

// -----

func.func @bad_wmma_width(%a: !wavemachine.reg<vgpr, 8>, %b: !wavemachine.reg<vgpr, 4>, %acc: !wavemachine.reg<vgpr, 8>) {
  // expected-error @below {{A operand must be !wavemachine.reg<vgpr, 4>}}
  %r = wavemachine.wmma_i32_16x16x16_iu8 %a, %b, %acc : (!wavemachine.reg<vgpr, 8>, !wavemachine.reg<vgpr, 4>, !wavemachine.reg<vgpr, 8>) -> !wavemachine.reg<vgpr, 8>
  return
}

// -----

func.func @missing_tuple_component(%offset: !wavemachine.reg<vgpr, 1>, %value: !wavemachine.reg<vgpr, 4>, %base: !wavemachine.reg<sgpr, 2>) {
  // expected-error @below {{requires a component attribute}}
  %t = wavemachine.global_store_tuple_b32 %offset, %value, %base : (!wavemachine.reg<vgpr, 1>, !wavemachine.reg<vgpr, 4>, !wavemachine.reg<sgpr, 2>) -> !wavemachine.mem.token
  return
}

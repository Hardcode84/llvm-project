// RUN: mlir-opt --waveamd-reg-alloc -split-input-file -verify-diagnostics %s

func.func @unsupported_register_class() {
  // expected-error @below {{waveamd-reg-alloc supports only SGPR(0) and VGPR(1) register classes}}
  %reg = wavemachine.arg {index = 0 : i64, pointer = false} : !wavemachine.reg<2, 1>
  return
}

// -----

// expected-error @below {{WaveMachine register allocator ran out of registers}}
func.func @too_many_vgprs() {
  %zero = wavemachine.imm 0 : !wavemachine.imm
  %reg = wavemachine.v_mov_b32_tuple %zero {registers = 33 : i64} : (!wavemachine.imm) -> !wavemachine.reg<1, 33>
  return
}

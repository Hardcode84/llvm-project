// RUN: mlir-opt --waveamd-reg-alloc -split-input-file -verify-diagnostics %s

module attributes {wavemachine.target = "amdgcn-amd-amdhsa--gfx1100"} {

func.func @unsupported_register_class() {
  // expected-error @below {{waveamd-reg-alloc supports only SGPR and VGPR register classes}}
  %reg = wavemachine.arg {index = 0 : i64, pointer = false} : !wavemachine.reg<agpr, 1>
  return
}

}

// -----

module attributes {wavemachine.target = "amdgcn-amd-amdhsa--gfx1100"} {

// expected-error @below {{WaveMachine register allocator ran out of registers}}
func.func @too_many_vgprs() {
  %zero = wavemachine.imm 0 : !wavemachine.imm
  %reg = wavemachine.v_mov_b32_tuple %zero {registers = 257 : i64} : (!wavemachine.imm) -> !wavemachine.reg<vgpr, 257>
  return
}

}

// -----

// expected-error @below {{waveamd-reg-alloc requires a wavemachine.target attribute}}
module {
  func.func @missing_target() {
    return
  }
}

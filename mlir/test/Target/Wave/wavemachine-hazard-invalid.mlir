// RUN: mlir-opt --waveamd-insert-hazard-waits -split-input-file -verify-diagnostics %s

module attributes {wavemachine.target = "amdgcn-amd-amdhsa--gfx1100"} {

func.func @kernel_arg_not_abi_lowered() attributes {wave.kernel} {
  // expected-error @below {{waveamd-insert-hazard-waits expects ABI-lowered kernel arguments}}
  %arg = wavemachine.arg {index = 0 : i64, pointer = false} : !wavemachine.reg<sgpr, 1>
  return
}

}

// -----

module attributes {wavemachine.target = "amdgcn-amd-amdhsa--gfx1100"} {

func.func @missing_smem_base() {
  %offset = wavemachine.imm 0 : !wavemachine.imm
  // expected-error @below {{'wavemachine.s_load_b32' op requires attribute 'base'}}
  %load = "wavemachine.s_load_b32"(%offset) : (!wavemachine.imm) -> !wavemachine.reg<sgpr, 1>
  return
}

}

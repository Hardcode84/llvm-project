// RUN: mlir-opt --wavemachine-insert-hazard-waits -split-input-file -verify-diagnostics %s

module attributes {wavemachine.target = "amdgcn-amd-amdhsa--gfx1100"} {

func.func @kernel_arg_not_abi_lowered() attributes {wave.kernel} {
  // expected-error @below {{wavemachine-insert-hazard-waits expects ABI-lowered kernel arguments}}
  %arg = "wavemachine.arg"() {index = 0 : i64, memref = false} : () -> !wavemachine.reg<0, 1>
  return
}

}

// -----

module attributes {wavemachine.target = "amdgcn-amd-amdhsa--gfx1100"} {

func.func @missing_smem_base() {
  %offset = "wavemachine.imm"() {value = 0 : i64} : () -> !wavemachine.imm
  // expected-error @below {{wavemachine-insert-hazard-waits expects scalar memory loads to carry a base register attribute}}
  %load = "wavemachine.s_load_b32"(%offset) : (!wavemachine.imm) -> !wavemachine.reg<0, 1>
  return
}

}

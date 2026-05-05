// RUN: mlir-opt --wavemachine-metadata -split-input-file -verify-diagnostics %s

module attributes {wavemachine.target = "amdgcn-amd-amdhsa--gfx1100"} {

// expected-error @below {{wavemachine-metadata requires ABI and resource attributes on kernels}}
func.func @missing_kernel_metadata_inputs() attributes {wave.kernel} {
  return
}

}

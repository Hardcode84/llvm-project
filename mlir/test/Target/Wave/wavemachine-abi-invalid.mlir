// RUN: mlir-opt --wavemachine-abi-lowering -split-input-file -verify-diagnostics %s

func.func @missing_memref_attr() attributes {wave.kernel} {
  // expected-error @below {{wavemachine-abi-lowering expects wavemachine.arg to have a memref attribute}}
  %arg = "wavemachine.arg"() {index = 0 : i64} : () -> !wavemachine.reg<0, 1>
  return
}

// -----

func.func @bad_kernel_arg_class() attributes {wave.kernel} {
  // expected-error @below {{wavemachine-abi-lowering expects kernel arguments to be SGPR WaveMachine registers}}
  %arg = "wavemachine.arg"() {index = 0 : i64, memref = false} : () -> !wavemachine.reg<1, 1>
  return
}

// -----

func.func @bad_kernel_arg_width() attributes {wave.kernel} {
  // expected-error @below {{wavemachine-abi-lowering found argument register width inconsistent with memref attribute}}
  %arg = "wavemachine.arg"() {index = 0 : i64, memref = true} : () -> !wavemachine.reg<0, 1>
  return
}

// -----

func.func @bad_arg_result_count() attributes {wave.kernel} {
  // expected-error @below {{wavemachine-abi-lowering expects wavemachine.arg to have one result}}
  "wavemachine.arg"() {index = 0 : i64, memref = false} : () -> ()
  return
}

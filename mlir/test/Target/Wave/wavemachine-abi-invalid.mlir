// RUN: mlir-opt --waveamd-abi-lowering -split-input-file -verify-diagnostics %s

func.func @missing_pointer_attr() attributes {wave.kernel} {
  // expected-error @below {{'wavemachine.arg' op requires attribute 'pointer'}}
  %arg = "wavemachine.arg"() {index = 0 : i64} : () -> !wavemachine.reg<sgpr, 1>
  return
}

// -----

func.func @bad_kernel_arg_class() attributes {wave.kernel} {
  // expected-error @below {{waveamd-abi-lowering expects kernel arguments to be SGPR WaveMachine registers}}
  %arg = wavemachine.arg {index = 0 : i64, pointer = false} : !wavemachine.reg<vgpr, 1>
  return
}

// -----

func.func @bad_kernel_arg_width() attributes {wave.kernel} {
  // expected-error @below {{waveamd-abi-lowering found argument register width inconsistent with pointer attribute}}
  %arg = wavemachine.arg {index = 0 : i64, pointer = true} : !wavemachine.reg<sgpr, 1>
  return
}

// -----

func.func @bad_arg_result_count() attributes {wave.kernel} {
  // expected-error @below {{'wavemachine.arg' op requires one result}}
  "wavemachine.arg"() {index = 0 : i64, pointer = false} : () -> ()
  return
}

// RUN: mlir-opt --waveamd-resource-info -split-input-file -verify-diagnostics %s

func.func @unallocated_register() {
  // expected-error @below {{waveamd-resource-info requires allocated register results}}
  %reg = wavemachine.v_mbcnt_lo : !wavemachine.reg<1, 1>
  return
}

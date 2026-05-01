// RUN: mlir-opt --wavemachine-resource-info -split-input-file -verify-diagnostics %s

func.func @unallocated_register() {
  // expected-error @below {{wavemachine-resource-info requires allocated register results}}
  %reg = "wavemachine.v_mbcnt_lo"() : () -> !wavemachine.reg<1, 1>
  return
}

// RUN: mlir-opt -verify-diagnostics %s

func.func @unknown_wavemachine_op() {
  // expected-error @below {{unregistered operation 'wavemachine.unknown_op' found in dialect ('wavemachine') that does not allow unknown operations}}
  "wavemachine.unknown_op"() : () -> ()
  return
}

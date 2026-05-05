// RUN: mlir-opt %s | FileCheck %s

// CHECK-LABEL: func.func @address_spaces
// CHECK-SAME: !wave.ptr<i32, #wave.global>
// CHECK-SAME: !wave.ptr<f32, #wave.shared>
// CHECK-SAME: !wave.ptr<i8, #wave.private>
func.func @address_spaces(
    %global: !wave.ptr<i32, #wave.global>,
    %shared: !wave.ptr<f32, #wave.shared>,
    %private: !wave.ptr<i8, #wave.private>) {
  return
}

// CHECK-LABEL: func.func @lane_varying_pointer
// CHECK-SAME: !wave.simd<!wave.ptr<i32, #wave.global>, 32>
func.func @lane_varying_pointer(
    %ptrs: !wave.simd<!wave.ptr<i32, #wave.global>, 32>) {
  return
}

// RUN: mlir-opt %s | FileCheck %s

// CHECK-LABEL: func.func @address_spaces
// CHECK-SAME: !wave.ptr<i32, #wave.global>
// CHECK-SAME: !wave.ptr<f32, #wave.shared>
// CHECK-SAME: !wave.ptr<i8, #wave.private>
// CHECK-SAME: !wave.ptr<i32, #waveamd.buffer>
func.func @address_spaces(
    %global: !wave.ptr<i32, #wave.global>,
    %shared: !wave.ptr<f32, #wave.shared>,
    %private: !wave.ptr<i8, #wave.private>,
    %buffer: !wave.ptr<i32, #waveamd.buffer>) {
  return
}

// CHECK-LABEL: func.func @lane_varying_pointer
// CHECK-SAME: !wave.simd<!wave.ptr<i32, #wave.global>, 32>
// CHECK-SAME: !wave.simd<!wave.ptr<i32, #waveamd.buffer>, 32>
func.func @lane_varying_pointer(
    %ptrs: !wave.simd<!wave.ptr<i32, #wave.global>, 32>,
    %buffer_ptrs: !wave.simd<!wave.ptr<i32, #waveamd.buffer>, 32>) {
  return
}

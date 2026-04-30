// RUN: mlir-opt %s -convert-wave-to-gpu | FileCheck %s

// CHECK-LABEL: func.func @lower_to_gpu
func.func @lower_to_gpu(%pred: i1, %value: i32) -> i32 {
  // CHECK: gpu.lane_id
  %lane = wave.lane_id
  // CHECK: gpu.subgroup_id
  %subgroup = wave.subgroup_id
  // CHECK: gpu.subgroup_size
  %size = wave.subgroup_size
  // CHECK: gpu.ballot {{.*}} : i32
  %mask = wave.ballot %pred : i32
  "test.consume"(%lane, %subgroup, %size, %mask) : (index, index, index, i32) -> ()
  // CHECK: gpu.subgroup_broadcast {{.*}}, first_active_lane : i32
  %first = wave.read_first %value : i32

  // CHECK: scf.if {{.*}} {
  wave.where %pred {
    // CHECK: "test.side_effect"
    "test.side_effect"() : () -> ()
    wave.yield
  } otherwise {
    // CHECK: "test.else_effect"
    "test.else_effect"() : () -> ()
    wave.yield
  }

  func.return %first : i32
}

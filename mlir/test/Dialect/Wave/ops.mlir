// RUN: mlir-opt %s | FileCheck %s
// RUN: mlir-opt %s | mlir-opt | FileCheck %s

// CHECK-LABEL: func.func @wave_ops
func.func @wave_ops(%pred: i1, %value: i32) -> i32 {
  // CHECK: wave.lane_id
  %lane = wave.lane_id
  // CHECK: wave.subgroup_id
  %subgroup = wave.subgroup_id
  // CHECK: wave.subgroup_size
  %size = wave.subgroup_size
  // CHECK: wave.ballot {{.*}} : i32
  %mask = wave.ballot %pred : i32
  // CHECK: wave.read_first {{.*}} : i32
  %first = wave.read_first %value : i32

  // CHECK: wave.where
  wave.where %pred {
    wave.yield
  } otherwise {
    wave.yield
  }

  func.return %first : i32
}

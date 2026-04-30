// RUN: mlir-opt %s -convert-wave-to-rocdl | FileCheck %s

// CHECK-LABEL: func.func @lower_to_rocdl
func.func @lower_to_rocdl(%pred: i1, %value: i32) -> i32 {
  // CHECK: rocdl.mbcnt.lo
  %lane = wave.lane_id
  // CHECK: rocdl.wave.id
  %subgroup = wave.subgroup_id
  // CHECK: rocdl.wavefrontsize
  %size = wave.subgroup_size
  // CHECK: rocdl.ballot {{.*}} : i32
  %mask = wave.ballot %pred : i32
  "test.consume"(%lane, %subgroup, %size, %mask) : (index, index, index, i32) -> ()
  // CHECK: rocdl.readfirstlane {{.*}} : i32
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

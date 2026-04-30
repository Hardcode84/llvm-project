// RUN: mlir-opt %s -convert-wave-to-rocdl | FileCheck %s

// CHECK-LABEL: func.func @lower_to_rocdl
func.func @lower_to_rocdl(%pred: i1, %value: i32, %out: memref<i32>) -> i32 {
  // CHECK: rocdl.mbcnt.lo
  %lane = wave.lane_id : !wave.simd<i32, 32>
  %vvalue = wave.splat %value : i32 -> !wave.simd<i32, 32>
  // CHECK: arith.addi
  %sum = wave.binary "addi" %lane, %vvalue : !wave.simd<i32, 32>, !wave.simd<i32, 32> -> !wave.simd<i32, 32>
  // CHECK: arith.cmpi
  %laneMask = wave.cmpi ult %lane, %vvalue : !wave.simd<i32, 32>, !wave.simd<i32, 32> -> !wave.mask<32>
  // CHECK: rocdl.wave.id
  %subgroup = wave.subgroup_id
  // CHECK: rocdl.wavefrontsize
  %size = wave.subgroup_size
  // CHECK: rocdl.ballot {{.*}} : i32
  %mask = wave.ballot %laneMask : !wave.mask<32> -> i32
  "test.consume"(%subgroup, %size, %mask) : (index, index, i32) -> ()
  // CHECK: rocdl.readfirstlane {{.*}} : i32
  %first = wave.read_first %sum : !wave.simd<i32, 32> -> i32
  // CHECK: memref.store
  wave.store %sum -> %out[] : (!wave.simd<i32, 32>, memref<i32>) -> ()

  // CHECK: scf.if {{.*}} {
  wave.where %laneMask {
    // CHECK: "test.side_effect"
    "test.side_effect"() : () -> ()
    wave.yield
  } otherwise {
    // CHECK: "test.else_effect"
    "test.else_effect"() : () -> ()
    wave.yield
  } : !wave.mask<32>

  func.return %first : i32
}

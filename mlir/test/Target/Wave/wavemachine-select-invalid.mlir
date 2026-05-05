// RUN: mlir-opt --waveamd-to-wavemachine -split-input-file -verify-diagnostics %s

func.func @unsupported_op(%x: i32, %y: i32) -> i32 {
  // expected-error @below {{unsupported operation in WaveMachine selection}}
  %sum = arith.addi %x, %y : i32
  return %sum : i32
}

// -----

func.func @unsupported_lane_id_width() {
  // expected-error @below {{WaveMachine backend supports only !wave.simd<i32, 32> lane_id}}
  %lane = wave.lane_id : !wave.simd<i32, 64>
  return
}

// -----

func.func @unsupported_binary_kind(%x: i32) {
  %lane = wave.lane_id : !wave.simd<i32, 32>
  %vx = wave.splat %x : i32 -> !wave.simd<i32, 32>
  // expected-error @below {{unsupported wave.binary kind}}
  %bad = wave.binary "subi" %lane, %vx : !wave.simd<i32, 32>, !wave.simd<i32, 32> -> !wave.simd<i32, 32>
  return
}

// -----

func.func @kernel_return_value(%x: i32) -> i32 attributes {wave.kernel} {
  // expected-error @below {{kernel functions must return void}}
  return %x : i32
}

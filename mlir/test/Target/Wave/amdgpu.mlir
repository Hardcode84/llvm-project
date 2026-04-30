// RUN: mlir-translate --wave-to-amdgpu-asm %s | FileCheck %s

// CHECK: .amdgcn_target "amdgcn-amd-amdhsa--gfx1100"
// CHECK-LABEL: wave_add:
func.func @wave_add(%x: i32) -> i32 {
  // CHECK: v_mbcnt_lo_u32_b32 [[LANE:v[0-9]+]], -1, 0
  %lane = wave.lane_id : !wave.simd<i32, 32>
  %vx = wave.splat %x : i32 -> !wave.simd<i32, 32>
  // CHECK: v_add_u32_e32 [[SUM:v[0-9]+]], [[LANE]], v0
  %sum = wave.binary "addi" %lane, %vx : !wave.simd<i32, 32>, !wave.simd<i32, 32> -> !wave.simd<i32, 32>
  // CHECK: v_readfirstlane_b32 s0, [[SUM]]
  %first = wave.read_first %sum : !wave.simd<i32, 32> -> i32
  // CHECK: s_setpc_b64 s[30:31]
  return %first : i32
}

// CHECK-LABEL: wave_where:
func.func @wave_where(%limit: i32) -> i32 {
  // CHECK: v_mbcnt_lo_u32_b32 [[LANE:v[0-9]+]], -1, 0
  %lane = wave.lane_id : !wave.simd<i32, 32>
  %vlimit = wave.splat %limit : i32 -> !wave.simd<i32, 32>
  // CHECK: v_cmp_lt_u32_e64 [[MASK:s[0-9]+]], [[LANE]], v0
  %active = wave.cmpi ult %lane, %vlimit : !wave.simd<i32, 32>, !wave.simd<i32, 32> -> !wave.mask<32>
  // CHECK: s_and_saveexec_b32 [[SAVE:s[0-9]+]], [[MASK]]
  // CHECK: s_cbranch_execz [[END:.Lwave_endif_[0-9]+]]
  wave.where %active {
    // CHECK: v_add_u32_e32
    %sum = wave.binary "addi" %lane, %vlimit : !wave.simd<i32, 32>, !wave.simd<i32, 32> -> !wave.simd<i32, 32>
    wave.yield
  } : !wave.mask<32>
  // CHECK: [[END]]:
  // CHECK: s_mov_b32 exec_lo, [[SAVE]]
  %bits = wave.ballot %active : !wave.mask<32> -> i32
  // CHECK: s_mov_b32 s0,
  return %bits : i32
}

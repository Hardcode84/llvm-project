// RUN: mlir-opt --waveamd-to-wavemachine %s | FileCheck %s --check-prefix=SELECT
// RUN: mlir-opt --waveamd-to-wavemachine --waveamd-abi-lowering --waveamd-reg-alloc %s | FileCheck %s --check-prefix=PIPELINE
// RUN: mlir-translate --wave-to-amdgpu-asm %s | FileCheck %s --check-prefix=ASM
// RUN: mlir-translate --wave-to-amdgpu-asm %s | llvm-mc -triple=amdgcn-amd-amdhsa -mcpu=gfx1100 -filetype=obj -o /dev/null

module attributes {wavemachine.target = "amdgcn-amd-amdhsa--gfx1100"} {

// SELECT-LABEL: func.func @buffer_store_kernel
// SELECT: wavemachine.arg {index = 0 : i64, pointer = true} : !wavemachine.reg<sgpr, 2>
// SELECT: wavemachine.make_buffer_rsrc
// SELECT: wavemachine.buffer_store_b32

// PIPELINE-LABEL: func.func @buffer_store_kernel
// PIPELINE: wavemachine.s_load_b64
// PIPELINE: wavemachine.make_buffer_rsrc
// PIPELINE: wavemachine.buffer_store_b32
// PIPELINE-SAME: !wavemachine.reg<sgpr, 4,

// ASM-LABEL: buffer_store_kernel:
// ASM: s_load_b64
// ASM: s_mov_b32
// ASM: buffer_store_dword {{v[0-9]+}}, {{v[0-9]+}}, {{s\[[0-9]+:[0-9]+\]}}, 0 offen
func.func @buffer_store_kernel(%out: !wave.ptr<i32, #wave.global>, %x: i32) attributes {wave.kernel} {
  %range = arith.constant 128 : i32
  %buffer = waveamd.make_buffer %out, %range : !wave.ptr<i32, #wave.global>, i32 -> !wave.ptr<i32, #waveamd.buffer>
  %lane = wave.lane_id : !wave.simd<i32, 32>
  %vx = wave.splat %x : i32 -> !wave.simd<i32, 32>
  %sum = wave.binary "addi" %lane, %vx : !wave.simd<i32, 32>, !wave.simd<i32, 32> -> !wave.simd<i32, 32>
  %ptrs = wave.ptr_add %buffer, %lane : !wave.ptr<i32, #waveamd.buffer>, !wave.simd<i32, 32> -> !wave.simd<!wave.ptr<i32, #waveamd.buffer>, 32>
  %store_token = wave.store %sum -> %ptrs : (!wave.simd<i32, 32>, !wave.simd<!wave.ptr<i32, #waveamd.buffer>, 32>) -> !wave.mem.token
  return
}

}

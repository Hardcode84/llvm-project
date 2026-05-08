// REQUIRES: host-supports-amdgpu
// RUN: mlir-translate --wave-to-amdgpu-asm %s | llvm-mc -triple=amdgcn-amd-amdhsa -mcpu=gfx1100 -filetype=obj -o %t.o
// RUN: ld.lld -shared %t.o -o %t.hsaco
// RUN: /opt/rocm/bin/hipcc %S/wave_buffer_runner.cpp -o %t.runner
// RUN: env LD_LIBRARY_PATH=/opt/rocm/lib %t.runner %t.hsaco | FileCheck %s --check-prefix=HW

module attributes {wavemachine.target = "amdgcn-amd-amdhsa--gfx1100"} {

// HW: buffer_store_kernel ok
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

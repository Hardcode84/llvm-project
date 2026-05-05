// RUN: mlir-opt --waveamd-to-wavemachine %s | FileCheck %s --check-prefix=SELECT
// RUN: mlir-opt --waveamd-to-wavemachine --waveamd-abi-lowering --waveamd-insert-ticket-waits %s | FileCheck %s --check-prefix=TICKET

module attributes {wavemachine.target = "amdgcn-amd-amdhsa--gfx1100"} {

// SELECT-LABEL: func.func @token_kernel
// SELECT: wavemachine.global_store_b32{{.*}} : {{.*}} -> !wavemachine.mem.token
// SELECT: wavemachine.token_join{{.*}} : (!wavemachine.mem.token) -> !wavemachine.mem.token
// SELECT: wavemachine.global_store_b32{{.*}} after {{.*}} : {{.*}} !wavemachine.mem.token) -> !wavemachine.mem.token
// SELECT: wavemachine.wait{{.*}} : (!wavemachine.mem.token) -> ()

// TICKET-LABEL: func.func @token_kernel
// TICKET: wavemachine.global_store_b32{{.*}} : {{.*}} -> !wavemachine.mem.token
// TICKET: wavemachine.s_waitcnt_vscnt
// TICKET-NEXT: wavemachine.global_store_b32{{.*}} after {{.*}} : {{.*}} !wavemachine.mem.token) -> !wavemachine.mem.token
// TICKET: wavemachine.s_waitcnt_vscnt
// TICKET-NEXT: wavemachine.wait
// TICKET-NOT: wavemachine.s_waitcnt_vscnt
// TICKET: wavemachine.s_endpgm
func.func @token_kernel(%out: !wave.ptr<i32, #wave.global>, %x: i32) attributes {wave.kernel} {
  %lane = wave.lane_id : !wave.simd<i32, 32>
  %vx = wave.splat %x : i32 -> !wave.simd<i32, 32>
  %ptrs = wave.ptr_add %out, %lane : !wave.ptr<i32, #wave.global>, !wave.simd<i32, 32> -> !wave.simd<!wave.ptr<i32, #wave.global>, 32>
  %t0 = wave.store %vx -> %ptrs : (!wave.simd<i32, 32>, !wave.simd<!wave.ptr<i32, #wave.global>, 32>) -> !wave.mem.token
  %t1 = wave.after %t0 : !wave.mem.token -> !wave.mem.token
  %t2 = wave.store %vx -> %ptrs after %t1 : (!wave.simd<i32, 32>, !wave.simd<!wave.ptr<i32, #wave.global>, 32>, !wave.mem.token) -> !wave.mem.token
  wave.wait %t2 : !wave.mem.token
  return
}

// SELECT-LABEL: func.func @join_kernel
// SELECT: wavemachine.token_join{{.*}} : (!wavemachine.mem.token, !wavemachine.mem.token) -> !wavemachine.mem.token
// SELECT: wavemachine.wait

// TICKET-LABEL: func.func @join_kernel
// TICKET: wavemachine.global_store_b32
// TICKET: wavemachine.global_store_b32
// TICKET: wavemachine.token_join
// TICKET: wavemachine.s_waitcnt_vscnt
// TICKET-NEXT: wavemachine.wait
func.func @join_kernel(%out: !wave.ptr<i32, #wave.global>, %x: i32) attributes {wave.kernel} {
  %lane = wave.lane_id : !wave.simd<i32, 32>
  %vx = wave.splat %x : i32 -> !wave.simd<i32, 32>
  %ptrs = wave.ptr_add %out, %lane : !wave.ptr<i32, #wave.global>, !wave.simd<i32, 32> -> !wave.simd<!wave.ptr<i32, #wave.global>, 32>
  %a = wave.store %vx -> %ptrs : (!wave.simd<i32, 32>, !wave.simd<!wave.ptr<i32, #wave.global>, 32>) -> !wave.mem.token
  %b = wave.store %vx -> %ptrs : (!wave.simd<i32, 32>, !wave.simd<!wave.ptr<i32, #wave.global>, 32>) -> !wave.mem.token
  %both = wave.join %a, %b : !wave.mem.token, !wave.mem.token -> !wave.mem.token
  wave.wait %both : !wave.mem.token
  return
}

}

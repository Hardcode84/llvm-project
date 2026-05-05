// RUN: mlir-opt --waveamd-to-wavemachine %s | FileCheck %s --check-prefix=SELECT
// RUN: mlir-opt --waveamd-to-wavemachine --waveamd-abi-lowering %s | FileCheck %s --check-prefix=ABI
// RUN: mlir-opt --waveamd-to-wavemachine --waveamd-abi-lowering --waveamd-insert-ticket-waits %s | FileCheck %s --check-prefix=TICKET
// RUN: mlir-opt --waveamd-to-wavemachine --waveamd-abi-lowering --waveamd-insert-ticket-waits --waveamd-insert-hazard-waits %s | FileCheck %s --check-prefix=HAZARD
// RUN: mlir-opt --waveamd-to-wavemachine --waveamd-abi-lowering --waveamd-insert-ticket-waits --waveamd-insert-hazard-waits --waveamd-reg-alloc %s | FileCheck %s --check-prefix=REGALLOC
// RUN: mlir-opt --waveamd-to-wavemachine --waveamd-abi-lowering --waveamd-insert-ticket-waits --waveamd-insert-hazard-waits --waveamd-reg-alloc --waveamd-resource-info %s | FileCheck %s --check-prefix=RESOURCE
// RUN: mlir-opt --waveamd-to-wavemachine --waveamd-abi-lowering --waveamd-insert-ticket-waits --waveamd-insert-hazard-waits --waveamd-reg-alloc --waveamd-resource-info --waveamd-metadata %s | FileCheck %s --check-prefix=METADATA

module attributes {wavemachine.target = "amdgcn-amd-amdhsa--gfx1100"} {

// SELECT-LABEL: func.func @where_test
// SELECT: wavemachine.arg {index = 0 : i64, pointer = false} : !wavemachine.reg<sgpr, 1>
// SELECT: wavemachine.v_mbcnt_lo : !wavemachine.reg<vgpr, 1>
// SELECT: wavemachine.v_cmp_lt_u32
// SELECT: wavemachine.s_and_saveexec_b32
// SELECT: wavemachine.s_cbranch_execz ".Lwave_where_test_endif_0"
// SELECT: wavemachine.label ".Lwave_where_test_endif_0"
// SELECT: wavemachine.s_mov_exec_lo
func.func @where_test(%limit: i32) -> i32 {
  %lane = wave.lane_id : !wave.simd<i32, 32>
  %vlimit = wave.splat %limit : i32 -> !wave.simd<i32, 32>
  %active = wave.cmpi ult %lane, %vlimit : !wave.simd<i32, 32>, !wave.simd<i32, 32> -> !wave.mask<32>
  wave.where %active {
    %sum = wave.binary "addi" %lane, %vlimit : !wave.simd<i32, 32>, !wave.simd<i32, 32> -> !wave.simd<i32, 32>
    wave.yield
  } : !wave.mask<32>
  %bits = wave.ballot %active : !wave.mask<32> -> i32
  return %bits : i32
}

// SELECT-LABEL: func.func @kernel_test
// SELECT: wavemachine.arg {index = 0 : i64, pointer = true} : !wavemachine.reg<sgpr, 2>
// SELECT: wavemachine.arg {index = 1 : i64, pointer = false} : !wavemachine.reg<sgpr, 1>
// SELECT: wavemachine.global_store_b32
// ABI-LABEL: func.func @kernel_test
// ABI: wavemachine.s_load_b64 {{.*}}, "s[0:1]"
// ABI: wavemachine.s_load_b32 {{.*}}, "s[0:1]"
// ABI-NOT: wavemachine.arg
// TICKET-LABEL: func.func @kernel_test
// TICKET: wavemachine.v_mbcnt_lo
// TICKET: wavemachine.s_waitcnt
// TICKET-NOT: wavemachine.s_delay_alu
// TICKET: wavemachine.v_add_u32
// TICKET: wavemachine.global_store_b32
// TICKET: wavemachine.s_waitcnt_vscnt
// TICKET: wavemachine.s_endpgm
// HAZARD-LABEL: func.func @kernel_test
// HAZARD: wavemachine.s_waitcnt
// HAZARD: wavemachine.s_delay_alu
// HAZARD: wavemachine.v_add_u32
// REGALLOC-LABEL: func.func @kernel_test
// REGALLOC: wavemachine.s_load_b64 {{.*}}, "s[0:1]" : (!wavemachine.imm) -> !wavemachine.reg<sgpr, 2, 2>
// REGALLOC: wavemachine.s_load_b32 {{.*}}, "s[0:1]" : (!wavemachine.imm) -> !wavemachine.reg<sgpr, 1, 4>
// REGALLOC: wavemachine.v_mbcnt_lo : !wavemachine.reg<vgpr, 1, 0>
// REGALLOC: wavemachine.v_add_u32{{.*}} -> !wavemachine.reg<vgpr, 1, 1>
// RESOURCE-LABEL: func.func @kernel_test
// RESOURCE-SAME: wavemachine.sgpr_count = 6 : i64
// RESOURCE-SAME: wavemachine.vgpr_count = 3 : i64
// METADATA: module attributes {wavemachine.target = "amdgcn-amd-amdhsa--gfx1100"}
// METADATA-LABEL: func.func @kernel_test
// METADATA-SAME: wavemachine.metadata
func.func @kernel_test(%out: !wave.ptr<i32, #wave.global>, %x: i32) attributes {wave.kernel} {
  %lane = wave.lane_id : !wave.simd<i32, 32>
  %vx = wave.splat %x : i32 -> !wave.simd<i32, 32>
  %sum = wave.binary "addi" %lane, %vx : !wave.simd<i32, 32>, !wave.simd<i32, 32> -> !wave.simd<i32, 32>
  %ptrs = wave.ptr_add %out, %lane : !wave.ptr<i32, #wave.global>, !wave.simd<i32, 32> -> !wave.simd<!wave.ptr<i32, #wave.global>, 32>
  %store_token = wave.store %sum -> %ptrs : (!wave.simd<i32, 32>, !wave.simd<!wave.ptr<i32, #wave.global>, 32>) -> !wave.mem.token
  return
}

}

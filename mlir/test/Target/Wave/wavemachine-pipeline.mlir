// RUN: mlir-opt --convert-wave-to-wavemachine %s | FileCheck %s --check-prefix=SELECT
// RUN: mlir-opt --convert-wave-to-wavemachine --wavemachine-abi-lowering %s | FileCheck %s --check-prefix=ABI
// RUN: mlir-opt --convert-wave-to-wavemachine --wavemachine-abi-lowering --wavemachine-insert-ticket-waits %s | FileCheck %s --check-prefix=TICKET
// RUN: mlir-opt --convert-wave-to-wavemachine --wavemachine-abi-lowering --wavemachine-insert-ticket-waits --wavemachine-insert-hazard-waits %s | FileCheck %s --check-prefix=HAZARD
// RUN: mlir-opt --convert-wave-to-wavemachine --wavemachine-abi-lowering --wavemachine-insert-ticket-waits --wavemachine-insert-hazard-waits --wavemachine-reg-alloc %s | FileCheck %s --check-prefix=REGALLOC
// RUN: mlir-opt --convert-wave-to-wavemachine --wavemachine-abi-lowering --wavemachine-insert-ticket-waits --wavemachine-insert-hazard-waits --wavemachine-reg-alloc --wavemachine-resource-info %s | FileCheck %s --check-prefix=RESOURCE
// RUN: mlir-opt --convert-wave-to-wavemachine --wavemachine-abi-lowering --wavemachine-insert-ticket-waits --wavemachine-insert-hazard-waits --wavemachine-reg-alloc --wavemachine-resource-info --wavemachine-metadata %s | FileCheck %s --check-prefix=METADATA

module attributes {wavemachine.target = "amdgcn-amd-amdhsa--gfx1100"} {

// SELECT-LABEL: func.func @where_test
// SELECT: "wavemachine.arg"() {index = 0 : i64, memref = false} : () -> !wavemachine.reg<0, 1>
// SELECT: "wavemachine.v_mbcnt_lo"() : () -> !wavemachine.reg<1, 1>
// SELECT: "wavemachine.v_cmp_lt_u32"
// SELECT: "wavemachine.s_and_saveexec_b32"
// SELECT: "wavemachine.s_cbranch_execz"() {label = ".Lwave_where_test_endif_0"}
// SELECT: "wavemachine.label"() {name = ".Lwave_where_test_endif_0"}
// SELECT: "wavemachine.s_mov_exec_lo"
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
// SELECT: "wavemachine.arg"() {index = 0 : i64, memref = true} : () -> !wavemachine.reg<0, 2>
// SELECT: "wavemachine.arg"() {index = 1 : i64, memref = false} : () -> !wavemachine.reg<0, 1>
// SELECT: "wavemachine.global_store_b32"
// ABI-LABEL: func.func @kernel_test
// ABI: "wavemachine.s_load_b64"{{.*}} {base = "s[0:1]"}
// ABI: "wavemachine.s_load_b32"{{.*}} {base = "s[0:1]"}
// ABI-NOT: "wavemachine.arg"
// TICKET-LABEL: func.func @kernel_test
// TICKET: "wavemachine.v_mbcnt_lo"
// TICKET: "wavemachine.s_waitcnt"
// TICKET-NOT: "wavemachine.s_delay_alu"
// TICKET: "wavemachine.v_add_u32"
// TICKET: "wavemachine.global_store_b32"
// TICKET: "wavemachine.s_waitcnt_vscnt"
// TICKET: "wavemachine.s_endpgm"
// HAZARD-LABEL: func.func @kernel_test
// HAZARD: "wavemachine.s_waitcnt"
// HAZARD: "wavemachine.s_delay_alu"
// HAZARD: "wavemachine.v_add_u32"
// REGALLOC-LABEL: func.func @kernel_test
// REGALLOC: "wavemachine.s_load_b64"{{.*}} {base = "s[0:1]", phys = 2 : i64}
// REGALLOC: "wavemachine.s_load_b32"{{.*}} {base = "s[0:1]", phys = 4 : i64}
// REGALLOC: "wavemachine.v_mbcnt_lo"() {phys = 0 : i64}
// REGALLOC: "wavemachine.v_add_u32"{{.*}} {phys = 1 : i64}
// RESOURCE-LABEL: func.func @kernel_test
// RESOURCE-SAME: wavemachine.sgpr_count = 6 : i64
// RESOURCE-SAME: wavemachine.vgpr_count = 3 : i64
// METADATA: module attributes {wavemachine.target = "amdgcn-amd-amdhsa--gfx1100"}
// METADATA-LABEL: func.func @kernel_test
// METADATA-SAME: wavemachine.metadata
func.func @kernel_test(%out: memref<32xi32>, %x: i32) attributes {wave.kernel} {
  %lane = wave.lane_id : !wave.simd<i32, 32>
  %vx = wave.splat %x : i32 -> !wave.simd<i32, 32>
  %sum = wave.binary "addi" %lane, %vx : !wave.simd<i32, 32>, !wave.simd<i32, 32> -> !wave.simd<i32, 32>
  wave.store %sum -> %out[%lane] : (!wave.simd<i32, 32>, memref<32xi32>, !wave.simd<i32, 32>) -> ()
  return
}

}

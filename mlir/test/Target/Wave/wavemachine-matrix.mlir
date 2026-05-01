// RUN: mlir-opt --convert-wave-to-wavemachine %s | FileCheck %s --check-prefix=SELECT
// RUN: mlir-opt --convert-wave-to-wavemachine --wavemachine-abi-lowering --wavemachine-insert-hazard-waits --wavemachine-reg-alloc --wavemachine-resource-info --wavemachine-metadata %s | FileCheck %s --check-prefix=PIPELINE
// RUN: mlir-translate --wave-to-amdgpu-asm %s | FileCheck %s --check-prefix=ASM
// RUN: mlir-translate --wave-to-amdgpu-asm %s | llvm-mc -triple=amdgcn-amd-amdhsa -mcpu=gfx1100 -filetype=obj -o /dev/null

// SELECT-LABEL: func.func @matrix_kernel
// SELECT: "wavemachine.v_mov_b32_tuple"{{.*}} : (!wavemachine.imm) -> !wavemachine.reg<1, 4>
// SELECT: "wavemachine.v_mov_b32_tuple"{{.*}} : (!wavemachine.imm) -> !wavemachine.reg<1, 4>
// SELECT: "wavemachine.v_mov_b32_tuple"{{.*}} : (!wavemachine.imm) -> !wavemachine.reg<1, 8>
// SELECT: "wavemachine.wmma_i32_16x16x16_iu8"{{.*}} : (!wavemachine.reg<1, 4>, !wavemachine.reg<1, 4>, !wavemachine.reg<1, 8>) -> !wavemachine.reg<1, 8>
// SELECT: "wavemachine.global_store_tuple_b32"{{.*}} {component = 0 : i64}
// SELECT: "wavemachine.global_store_tuple_b32"{{.*}} {component = 7 : i64}

// PIPELINE: module attributes {wavemachine.target = "amdgcn-amd-amdhsa--gfx1100"}
// PIPELINE-LABEL: func.func @matrix_kernel
// PIPELINE-SAME: wavemachine.metadata
// PIPELINE-SAME: wavemachine.vgpr_count
// PIPELINE: "wavemachine.wmma_i32_16x16x16_iu8"{{.*}} {phys =
// PIPELINE: "wavemachine.global_store_tuple_b32"{{.*}} {component = 7 : i64}

// ASM-LABEL: matrix_kernel:
// ASM: s_load_b64 [[OUT:s\[[0-9]+:[0-9]+\]]], s[0:1], 0x0
// ASM: v_mov_b32 [[A0:v[0-9]+]], 0
// ASM: v_mov_b32 [[B0:v[0-9]+]], 0
// ASM: v_mov_b32 [[C0:v[0-9]+]], 7
// ASM: v_wmma_i32_16x16x16_iu8 [[DST:v\[[0-9]+:[0-9]+\]]], [[A:v\[[0-9]+:[0-9]+\]]], [[B:v\[[0-9]+:[0-9]+\]]], [[C:v\[[0-9]+:[0-9]+\]]]
// ASM: global_store_b32 {{v[0-9]+}}, {{v[0-9]+}}, [[OUT]]
// ASM: s_waitcnt vmcnt(0)
// ASM: s_endpgm
func.func @matrix_kernel(%out: memref<256xi32>) attributes {wave.kernel} {
  %zero = arith.constant 0 : i32
  %seven = arith.constant 7 : i32
  %base = arith.constant 0 : index
  %a = wave.fragment_fill %zero : i32 -> !wave.fragment<0, i8, 16, 16, 32, 4>
  %b = wave.fragment_fill %zero : i32 -> !wave.fragment<1, i8, 16, 16, 32, 4>
  %acc = wave.fragment_fill %seven : i32 -> !wave.fragment<2, i32, 16, 16, 32, 8>
  %result = wave.mma "wmma.i32.16x16x16.iu8" %a, %b, %acc : !wave.fragment<0, i8, 16, 16, 32, 4>, !wave.fragment<1, i8, 16, 16, 32, 4>, !wave.fragment<2, i32, 16, 16, 32, 8> -> !wave.fragment<2, i32, 16, 16, 32, 8>
  wave.fragment_store %result -> %out[%base] : (!wave.fragment<2, i32, 16, 16, 32, 8>, memref<256xi32>, index) -> ()
  return
}

// SELECT-LABEL: func.func @matrix_f16_kernel
// SELECT: "wavemachine.v_mov_b32_tuple"{{.*}} : (!wavemachine.imm) -> !wavemachine.reg<1, 8>
// SELECT: "wavemachine.v_mov_b32_tuple"{{.*}} : (!wavemachine.imm) -> !wavemachine.reg<1, 8>
// SELECT: "wavemachine.v_mov_b32_tuple"{{.*}} : (!wavemachine.imm) -> !wavemachine.reg<1, 8>
// SELECT: "wavemachine.wmma_f32_16x16x16_f16"{{.*}} : (!wavemachine.reg<1, 8>, !wavemachine.reg<1, 8>, !wavemachine.reg<1, 8>) -> !wavemachine.reg<1, 8>

// PIPELINE-LABEL: func.func @matrix_f16_kernel
// PIPELINE-SAME: wavemachine.metadata
// PIPELINE: "wavemachine.wmma_f32_16x16x16_f16"{{.*}} {phys =

// ASM-LABEL: matrix_f16_kernel:
// ASM: v_wmma_f32_16x16x16_f16 [[DST:v\[[0-9]+:[0-9]+\]]], [[A:v\[[0-9]+:[0-9]+\]]], [[B:v\[[0-9]+:[0-9]+\]]], [[C:v\[[0-9]+:[0-9]+\]]]
// ASM: global_store_b32 {{v[0-9]+}}, {{v[0-9]+}}, {{s\[[0-9]+:[0-9]+\]}} offset:28
// ASM: s_endpgm
func.func @matrix_f16_kernel(%out: memref<256xi32>) attributes {wave.kernel} {
  %zero = arith.constant 0 : i32
  %seven_as_f32_bits = arith.constant 1088421888 : i32
  %base = arith.constant 0 : index
  %a = wave.fragment_fill %zero : i32 -> !wave.fragment<0, f16, 16, 16, 32, 8>
  %b = wave.fragment_fill %zero : i32 -> !wave.fragment<1, f16, 16, 16, 32, 8>
  %acc = wave.fragment_fill %seven_as_f32_bits : i32 -> !wave.fragment<2, f32, 16, 16, 32, 8>
  %result = wave.mma "wmma.f32.16x16x16.f16" %a, %b, %acc : !wave.fragment<0, f16, 16, 16, 32, 8>, !wave.fragment<1, f16, 16, 16, 32, 8>, !wave.fragment<2, f32, 16, 16, 32, 8> -> !wave.fragment<2, f32, 16, 16, 32, 8>
  wave.fragment_store %result -> %out[%base] : (!wave.fragment<2, f32, 16, 16, 32, 8>, memref<256xi32>, index) -> ()
  return
}

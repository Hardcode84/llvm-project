// RUN: mlir-opt %s \
// RUN: | mlir-opt -convert-wave-to-rocdl -convert-scf-to-cf \
// RUN: | mlir-opt -gpu-kernel-outlining \
// RUN: | mlir-opt -pass-pipeline='builtin.module(gpu.module(strip-debuginfo,convert-gpu-to-rocdl{use-bare-ptr-memref-call-conv=true}),rocdl-attach-target{chip=%chip})' \
// RUN: | mlir-opt -gpu-to-llvm=use-bare-pointers-for-kernels=true -reconcile-unrealized-casts -gpu-module-to-binary \
// RUN: | mlir-runner \
// RUN:   --shared-libs=%mlir_rocm_runtime \
// RUN:   --shared-libs=%mlir_runner_utils \
// RUN:   --entry-point-result=void \
// RUN: | FileCheck %s

func.func @write_lane_ids(%dst : memref<32xi32>) {
  %c1 = arith.constant 1 : index
  %c32 = arith.constant 32 : index
  gpu.launch blocks(%bx, %by, %bz) in (%grid_x = %c1, %grid_y = %c1, %grid_z = %c1)
             threads(%tx, %ty, %tz) in (%block_x = %c32, %block_y = %c1, %block_z = %c1) {
    %lane = wave.lane_id : !wave.simd<i32, 32>
    wave.store %lane -> %dst[%tx] : (!wave.simd<i32, 32>, memref<32xi32>, index) -> ()
    gpu.terminator
  }
  return
}

func.func @write_masked_values(%dst : memref<32xi32>) {
  %c1 = arith.constant 1 : index
  %c32 = arith.constant 32 : index
  %c8_i32 = arith.constant 8 : i32
  %c100_i32 = arith.constant 100 : i32
  gpu.launch blocks(%bx, %by, %bz) in (%grid_x = %c1, %grid_y = %c1, %grid_z = %c1)
             threads(%tx, %ty, %tz) in (%block_x = %c32, %block_y = %c1, %block_z = %c1) {
    %lane = wave.lane_id : !wave.simd<i32, 32>
    %limit = wave.splat %c8_i32 : i32 -> !wave.simd<i32, 32>
    %base = wave.splat %c100_i32 : i32 -> !wave.simd<i32, 32>
    %active = wave.cmpi ult %lane, %limit : !wave.simd<i32, 32>, !wave.simd<i32, 32> -> !wave.mask<32>
    %value = wave.binary "addi" %lane, %base : !wave.simd<i32, 32>, !wave.simd<i32, 32> -> !wave.simd<i32, 32>
    wave.where %active {
      wave.store %value -> %dst[%tx] : (!wave.simd<i32, 32>, memref<32xi32>, index) -> ()
      wave.yield
    } : !wave.mask<32>
    gpu.terminator
  }
  return
}

// CHECK: [0, 1, 2, 3, 4, 5, 6, 7
// CHECK-SAME: 8, 9, 10, 11, 12, 13, 14, 15
// CHECK-SAME: 16, 17, 18, 19, 20, 21, 22, 23
// CHECK-SAME: 24, 25, 26, 27, 28, 29, 30, 31]
// CHECK: [100, 101, 102, 103, 104, 105, 106, 107
// CHECK-SAME: 0, 0, 0, 0, 0, 0, 0, 0
// CHECK-SAME: 0, 0, 0, 0, 0, 0, 0, 0
// CHECK-SAME: 0, 0, 0, 0, 0, 0, 0, 0]
func.func @main() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c32 = arith.constant 32 : index
  %zero = arith.constant 0 : i32

  %laneStorage = memref.alloc() : memref<32xi32>
  %maskedStorage = memref.alloc() : memref<32xi32>
  %lanes = memref.cast %laneStorage : memref<32xi32> to memref<?xi32>
  %masked = memref.cast %maskedStorage : memref<32xi32> to memref<?xi32>

  scf.for %i = %c0 to %c32 step %c1 {
    memref.store %zero, %lanes[%i] : memref<?xi32>
    memref.store %zero, %masked[%i] : memref<?xi32>
  }

  %lanesUnranked = memref.cast %lanes : memref<?xi32> to memref<*xi32>
  %maskedUnranked = memref.cast %masked : memref<?xi32> to memref<*xi32>
  gpu.host_register %lanesUnranked : memref<*xi32>
  gpu.host_register %maskedUnranked : memref<*xi32>

  %lanesDevice = call @mgpuMemGetDeviceMemRef1dInt32(%lanes) : (memref<?xi32>) -> (memref<?xi32>)
  %maskedDevice = call @mgpuMemGetDeviceMemRef1dInt32(%masked) : (memref<?xi32>) -> (memref<?xi32>)
  %lanesDeviceStatic = memref.cast %lanesDevice : memref<?xi32> to memref<32xi32>
  %maskedDeviceStatic = memref.cast %maskedDevice : memref<?xi32> to memref<32xi32>

  call @write_lane_ids(%lanesDeviceStatic) : (memref<32xi32>) -> ()
  call @write_masked_values(%maskedDeviceStatic) : (memref<32xi32>) -> ()

  call @printMemrefI32(%lanesUnranked) : (memref<*xi32>) -> ()
  call @printMemrefI32(%maskedUnranked) : (memref<*xi32>) -> ()
  return
}

func.func private @mgpuMemGetDeviceMemRef1dInt32(%ptr : memref<?xi32>) -> (memref<?xi32>)
func.func private @printMemrefI32(%ptr : memref<*xi32>)

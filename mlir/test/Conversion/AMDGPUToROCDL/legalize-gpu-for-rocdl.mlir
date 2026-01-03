// RUN: mlir-opt %s --legalize-gpu-for-rocdl | FileCheck %s

// CHECK-LABEL: func @gpu_shuffle_i64
// CHECK-SAME: (%[[ARG0:.*]]: i64, %[[OFFSET:.*]]: i32, %[[WIDTH:.*]]: i32)
func.func @gpu_shuffle_i64(%arg0: i64, %offset: i32, %width: i32) -> i64 {
  // CHECK: %[[CAST:.*]] = llvm.bitcast %[[ARG0]] : i64 to vector<2xi32>
  // CHECK: %[[ELEMS:.*]]:2 = vector.to_elements %[[CAST]] : vector<2xi32>
  // CHECK: %[[SHFL0:.*]], %{{.*}} = gpu.shuffle xor %[[ELEMS]]#0, %[[OFFSET]], %[[WIDTH]] : i32
  // CHECK: %[[SHFL1:.*]], %{{.*}} = gpu.shuffle xor %[[ELEMS]]#1, %[[OFFSET]], %[[WIDTH]] : i32
  // CHECK: %[[VEC:.*]] = vector.from_elements %[[SHFL0]], %[[SHFL1]] : vector<2xi32>
  // CHECK: %[[RESULT:.*]] = llvm.bitcast %[[VEC]] : vector<2xi32> to i64
  %result, %valid = gpu.shuffle xor %arg0, %offset, %width : i64
  // CHECK: return %[[RESULT]] : i64
  return %result : i64
}

// CHECK-LABEL: func @gpu_shuffle_vector
// CHECK-SAME: (%[[ARG0:.*]]: vector<4xf16>, %[[OFFSET:.*]]: i32, %[[WIDTH:.*]]: i32)
func.func @gpu_shuffle_vector(%arg0: vector<4xf16>, %offset: i32, %width: i32) -> vector<4xf16> {
  // CHECK: %[[CAST:.*]] = llvm.bitcast %[[ARG0]] : vector<4xf16> to vector<2xi32>
  // CHECK: %[[ELEMS:.*]]:2 = vector.to_elements %[[CAST]] : vector<2xi32>
  // CHECK: %[[SHFL0:.*]], %{{.*}} = gpu.shuffle xor %[[ELEMS]]#0, %[[OFFSET]], %[[WIDTH]] : i32
  // CHECK: %[[SHFL1:.*]], %{{.*}} = gpu.shuffle xor %[[ELEMS]]#1, %[[OFFSET]], %[[WIDTH]] : i32
  // CHECK: %[[VEC:.*]] = vector.from_elements %[[SHFL0]], %[[SHFL1]] : vector<2xi32>
  // CHECK: %[[RESULT:.*]] = llvm.bitcast %[[VEC]] : vector<2xi32> to vector<4xf16>
  %result, %valid = gpu.shuffle xor %arg0, %offset, %width : vector<4xf16>
  // CHECK: return %[[RESULT]] : vector<4xf16>
  return %result : vector<4xf16>
}

// CHECK-LABEL: func @gpu_shuffle_i32_no_decompose
// CHECK-SAME: (%[[ARG0:.*]]: i32, %[[OFFSET:.*]]: i32, %[[WIDTH:.*]]: i32)
func.func @gpu_shuffle_i32_no_decompose(%arg0: i32, %offset: i32, %width: i32) -> i32 {
  // CHECK-NOT: llvm.bitcast
  // CHECK: %[[RESULT:.*]], %{{.*}} = gpu.shuffle xor %[[ARG0]], %[[OFFSET]], %[[WIDTH]] : i32
  %result, %valid = gpu.shuffle xor %arg0, %offset, %width : i32
  // CHECK: return %[[RESULT]] : i32
  return %result : i32
}

// CHECK-LABEL: func @amdgpu_swizzle_i64
// CHECK-SAME: (%[[ARG0:.*]]: i64)
func.func @amdgpu_swizzle_i64(%arg0: i64) -> i64 {
  // CHECK: %[[CAST:.*]] = llvm.bitcast %[[ARG0]] : i64 to vector<2xi32>
  // CHECK: %[[ELEMS:.*]]:2 = vector.to_elements %[[CAST]] : vector<2xi32>
  // CHECK: %[[SWIZ0:.*]] = amdgpu.swizzle_bitmode %[[ELEMS]]#0 15 0 1 : i32
  // CHECK: %[[SWIZ1:.*]] = amdgpu.swizzle_bitmode %[[ELEMS]]#1 15 0 1 : i32
  // CHECK: %[[VEC:.*]] = vector.from_elements %[[SWIZ0]], %[[SWIZ1]] : vector<2xi32>
  // CHECK: %[[RESULT:.*]] = llvm.bitcast %[[VEC]] : vector<2xi32> to i64
  %result = amdgpu.swizzle_bitmode %arg0 15 0 1 : i64
  // CHECK: return %[[RESULT]] : i64
  return %result : i64
}

// CHECK-LABEL: func @amdgpu_swizzle_vector
// CHECK-SAME: (%[[ARG0:.*]]: vector<2xf32>)
func.func @amdgpu_swizzle_vector(%arg0: vector<2xf32>) -> vector<2xf32> {
  // CHECK: %[[CAST:.*]] = llvm.bitcast %[[ARG0]] : vector<2xf32> to vector<2xi32>
  // CHECK: %[[ELEMS:.*]]:2 = vector.to_elements %[[CAST]] : vector<2xi32>
  // CHECK: %[[SWIZ0:.*]] = amdgpu.swizzle_bitmode %[[ELEMS]]#0 31 0 0 : i32
  // CHECK: %[[SWIZ1:.*]] = amdgpu.swizzle_bitmode %[[ELEMS]]#1 31 0 0 : i32
  // CHECK: %[[VEC:.*]] = vector.from_elements %[[SWIZ0]], %[[SWIZ1]] : vector<2xi32>
  // CHECK: %[[RESULT:.*]] = llvm.bitcast %[[VEC]] : vector<2xi32> to vector<2xf32>
  %result = amdgpu.swizzle_bitmode %arg0 31 0 0 : vector<2xf32>
  // CHECK: return %[[RESULT]] : vector<2xf32>
  return %result : vector<2xf32>
}

// CHECK-LABEL: func @amdgpu_permlane_i64
// CHECK-SAME: (%[[ARG0:.*]]: i64)
func.func @amdgpu_permlane_i64(%arg0: i64) -> i64 {
  // CHECK: %[[CAST:.*]] = llvm.bitcast %[[ARG0]] : i64 to vector<2xi32>
  // CHECK: %[[ELEMS:.*]]:2 = vector.to_elements %[[CAST]] : vector<2xi32>
  // CHECK: %[[PERM0:.*]] = amdgpu.permlane_swap %[[ELEMS]]#0 16 : i32
  // CHECK: %[[PERM1:.*]] = amdgpu.permlane_swap %[[ELEMS]]#1 16 : i32
  // CHECK: %[[VEC:.*]] = vector.from_elements %[[PERM0]], %[[PERM1]] : vector<2xi32>
  // CHECK: %[[RESULT:.*]] = llvm.bitcast %[[VEC]] : vector<2xi32> to i64
  %result = amdgpu.permlane_swap %arg0 16 : i64
  // CHECK: return %[[RESULT]] : i64
  return %result : i64
}

// CHECK-LABEL: func @amdgpu_swizzle_i32_no_decompose
// CHECK-SAME: (%[[ARG0:.*]]: i32)
func.func @amdgpu_swizzle_i32_no_decompose(%arg0: i32) -> i32 {
  // CHECK-NOT: llvm.bitcast
  // CHECK: %[[RESULT:.*]] = amdgpu.swizzle_bitmode %[[ARG0]] 15 0 1 : i32
  %result = amdgpu.swizzle_bitmode %arg0 15 0 1 : i32
  // CHECK: return %[[RESULT]] : i32
  return %result : i32
}

// CHECK-LABEL: func @gpu_subgroup_broadcast_vector
// CHECK-SAME: (%[[ARG0:.*]]: vector<2xf32>, %[[LANE:.*]]: i32)
func.func @gpu_subgroup_broadcast_vector(%arg0: vector<2xf32>, %lane: i32) -> vector<2xf32> {
  // CHECK: %[[CAST:.*]] = llvm.bitcast %[[ARG0]] : vector<2xf32> to vector<2xi32>
  // CHECK: %[[ELEMS:.*]]:2 = vector.to_elements %[[CAST]] : vector<2xi32>
  // CHECK: %[[BCAST0:.*]] = gpu.subgroup_broadcast %[[ELEMS]]#0, specific_lane %[[LANE]] : i32
  // CHECK: %[[BCAST1:.*]] = gpu.subgroup_broadcast %[[ELEMS]]#1, specific_lane %[[LANE]] : i32
  // CHECK: %[[VEC:.*]] = vector.from_elements %[[BCAST0]], %[[BCAST1]] : vector<2xi32>
  // CHECK: %[[RESULT:.*]] = llvm.bitcast %[[VEC]] : vector<2xi32> to vector<2xf32>
  %result = gpu.subgroup_broadcast %arg0, specific_lane %lane : vector<2xf32>
  // CHECK: return %[[RESULT]] : vector<2xf32>
  return %result : vector<2xf32>
}

// CHECK-LABEL: func @gpu_subgroup_broadcast_first_active
// CHECK-SAME: (%[[ARG0:.*]]: vector<4xf16>)
func.func @gpu_subgroup_broadcast_first_active(%arg0: vector<4xf16>) -> vector<4xf16> {
  // CHECK: %[[CAST:.*]] = llvm.bitcast %[[ARG0]] : vector<4xf16> to vector<2xi32>
  // CHECK: %[[ELEMS:.*]]:2 = vector.to_elements %[[CAST]] : vector<2xi32>
  // CHECK: %[[BCAST0:.*]] = gpu.subgroup_broadcast %[[ELEMS]]#0, first_active_lane : i32
  // CHECK: %[[BCAST1:.*]] = gpu.subgroup_broadcast %[[ELEMS]]#1, first_active_lane : i32
  // CHECK: %[[VEC:.*]] = vector.from_elements %[[BCAST0]], %[[BCAST1]] : vector<2xi32>
  // CHECK: %[[RESULT:.*]] = llvm.bitcast %[[VEC]] : vector<2xi32> to vector<4xf16>
  %result = gpu.subgroup_broadcast %arg0, first_active_lane : vector<4xf16>
  // CHECK: return %[[RESULT]] : vector<4xf16>
  return %result : vector<4xf16>
}

// CHECK-LABEL: func @gpu_subgroup_broadcast_i32_no_decompose
// CHECK-SAME: (%[[ARG0:.*]]: i32, %[[LANE:.*]]: i32)
func.func @gpu_subgroup_broadcast_i32_no_decompose(%arg0: i32, %lane: i32) -> i32 {
  // CHECK-NOT: llvm.bitcast
  // CHECK: %[[RESULT:.*]] = gpu.subgroup_broadcast %[[ARG0]], specific_lane %[[LANE]] : i32
  %result = gpu.subgroup_broadcast %arg0, specific_lane %lane : i32
  // CHECK: return %[[RESULT]] : i32
  return %result : i32
}

// CHECK-LABEL: func @gpu_subgroup_broadcast_i8
// CHECK-SAME: (%[[ARG0:.*]]: i8, %[[LANE:.*]]: i32)
func.func @gpu_subgroup_broadcast_i8(%arg0: i8, %lane: i32) -> i8 {
  // CHECK: %[[EXT:.*]] = llvm.zext %[[ARG0]] : i8 to i32
  // CHECK: %[[BCAST:.*]] = gpu.subgroup_broadcast %[[EXT]], specific_lane %[[LANE]] : i32
  // CHECK: %[[TRUNC:.*]] = llvm.trunc %[[BCAST]] : i32 to i8
  %result = gpu.subgroup_broadcast %arg0, specific_lane %lane : i8
  // CHECK: return %[[TRUNC]] : i8
  return %result : i8
}

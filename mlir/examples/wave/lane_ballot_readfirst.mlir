// Compile with:
//   mlir-opt lane_ballot_readfirst.mlir \
//     --pass-pipeline='builtin.module(convert-wave-to-rocdl,convert-scf-to-cf,convert-cf-to-llvm,convert-arith-to-llvm,convert-index-to-llvm,convert-func-to-llvm,reconcile-unrealized-casts)' \
//   | mlir-translate --mlir-to-llvmir \
//   | llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1100 -filetype=asm -o -

module attributes {llvm.target_triple = "amdgcn-amd-amdhsa"} {
  func.func @lane_ballot_readfirst(%limit: i32) -> i32 {
    %lane = wave.lane_id : !wave.simd<i32, 32>
    %vlimit = wave.splat %limit : i32 -> !wave.simd<i32, 32>
    %active = wave.cmpi ult %lane, %vlimit : !wave.simd<i32, 32>, !wave.simd<i32, 32> -> !wave.mask<32>
    %mask = wave.ballot %active : !wave.mask<32> -> i32
    %first = wave.read_first %lane : !wave.simd<i32, 32> -> i32
    %out = arith.xori %first, %mask : i32
    return %out : i32
  }
}

// Compile with:
//   mlir-opt scf_loop_masks.mlir \
//     --pass-pipeline='builtin.module(convert-wave-to-rocdl,convert-scf-to-cf,convert-cf-to-llvm,convert-arith-to-llvm,convert-index-to-llvm,convert-func-to-llvm,reconcile-unrealized-casts)' \
//   | mlir-translate --mlir-to-llvmir \
//   | llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1100 -filetype=asm -o -

module attributes {llvm.target_triple = "amdgcn-amd-amdhsa"} {
  func.func @scf_loop_masks(%seed: i32) -> i32 {
    %lane = wave.lane_id : !wave.simd<i32, 32>
    %c0 = arith.constant 0 : index
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index
    %zero = arith.constant 0 : i32
    %one = arith.constant 1 : i32

    %acc = scf.for %i = %c0 to %c4 step %c1 iter_args(%carry = %zero) -> (i32) {
      %i32 = arith.index_cast %i : index to i32
      %shift = arith.shli %one, %i32 : i32
      %vshift = wave.splat %shift : i32 -> !wave.simd<i32, 32>
      %vzero = wave.splat %zero : i32 -> !wave.simd<i32, 32>
      %maskedLane = wave.binary "andi" %lane, %vshift : !wave.simd<i32, 32>, !wave.simd<i32, 32> -> !wave.simd<i32, 32>
      %pred = wave.cmpi eq %maskedLane, %vzero : !wave.simd<i32, 32>, !wave.simd<i32, 32> -> !wave.mask<32>
      %ballot = wave.ballot %pred : !wave.mask<32> -> i32
      %mixed = arith.xori %ballot, %seed : i32
      %next = arith.addi %carry, %mixed : i32
      scf.yield %next : i32
    }

    %vacc = wave.splat %acc : i32 -> !wave.simd<i32, 32>
    %uniform = wave.read_first %vacc : !wave.simd<i32, 32> -> i32
    return %uniform : i32
  }
}

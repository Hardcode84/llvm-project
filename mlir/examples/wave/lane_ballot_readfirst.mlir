// Compile with:
//   mlir-opt lane_ballot_readfirst.mlir \
//     --pass-pipeline='builtin.module(convert-wave-to-rocdl,convert-scf-to-cf,convert-cf-to-llvm,convert-arith-to-llvm,convert-index-to-llvm,convert-func-to-llvm,reconcile-unrealized-casts)' \
//   | mlir-translate --mlir-to-llvmir \
//   | llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1100 -filetype=asm -o -

module attributes {llvm.target_triple = "amdgcn-amd-amdhsa"} {
  func.func @lane_ballot_readfirst(%limit: i32) -> i32 {
    %lane = wave.lane_id
    %lane32 = arith.index_cast %lane : index to i32
    %active = arith.cmpi ult, %lane32, %limit : i32
    %mask = wave.ballot %active : i32
    %first = wave.read_first %mask : i32
    return %first : i32
  }
}

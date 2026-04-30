// Compile with:
//   mlir-opt where_store.mlir \
//     --pass-pipeline='builtin.module(convert-wave-to-rocdl,convert-scf-to-cf,convert-cf-to-llvm,convert-arith-to-llvm,convert-index-to-llvm,reconcile-unrealized-casts)' \
//   | mlir-translate --mlir-to-llvmir \
//   | llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1100 -filetype=asm -o -

module attributes {llvm.target_triple = "amdgcn-amd-amdhsa"} {
  llvm.func @where_store(%ptr: !llvm.ptr, %value: i32) {
    %lane = wave.lane_id
    %lane32 = arith.index_cast %lane : index to i32
    %c4 = llvm.mlir.constant(4 : i32) : i32
    %laneMod4 = llvm.urem %lane32, %c4 : i32
    %zero = llvm.mlir.constant(0 : i32) : i32
    %selected = llvm.icmp "eq" %laneMod4, %zero : i32

    wave.where %selected {
      llvm.store %value, %ptr : i32, !llvm.ptr
      wave.yield
    } otherwise {
      wave.yield
    }

    llvm.return
  }
}

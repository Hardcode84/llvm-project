// Compile with:
//   mlir-opt where_store.mlir \
//     --pass-pipeline='builtin.module(convert-wave-to-rocdl,convert-scf-to-cf,convert-cf-to-llvm,convert-arith-to-llvm,convert-index-to-llvm,reconcile-unrealized-casts)' \
//   | mlir-translate --mlir-to-llvmir \
//   | llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1100 -filetype=asm -o -

module attributes {llvm.target_triple = "amdgcn-amd-amdhsa"} {
  llvm.func @where_store(%ptr: !llvm.ptr, %value: i32) {
    %lane = wave.lane_id : !wave.simd<i32, 32>
    %four = llvm.mlir.constant(4 : i32) : i32
    %vfour = wave.splat %four : i32 -> !wave.simd<i32, 32>
    %selected = wave.cmpi ult %lane, %vfour : !wave.simd<i32, 32>, !wave.simd<i32, 32> -> !wave.mask<32>

    wave.where %selected {
      llvm.store %value, %ptr : i32, !llvm.ptr
      wave.yield
    } otherwise {
      wave.yield
    } : !wave.mask<32>

    llvm.return
  }
}

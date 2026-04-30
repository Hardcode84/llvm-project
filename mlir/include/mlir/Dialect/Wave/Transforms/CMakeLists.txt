set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name Wave)
add_mlir_dialect_tablegen_target(MLIRWaveTransformsIncGen)

add_mlir_doc(Passes WavePasses ./ -gen-pass-doc)

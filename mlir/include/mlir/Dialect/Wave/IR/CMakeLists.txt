add_mlir_dialect(WaveOps wave)
add_mlir_doc(WaveOps WaveOps Dialects/ -gen-dialect-doc)

set(LLVM_TARGET_DEFINITIONS WaveAttrs.td)
mlir_tablegen(WaveOpsAttributes.h.inc -gen-attrdef-decls -attrdefs-dialect=wave)
mlir_tablegen(WaveOpsAttributes.cpp.inc -gen-attrdef-defs -attrdefs-dialect=wave)
add_mlir_dialect_tablegen_target(MLIRWaveAttributesIncGen)

add_mlir_dialect(WaveAMDOps waveamd)
add_mlir_doc(WaveAMDOps WaveAMDOps Dialects/ -gen-dialect-doc)

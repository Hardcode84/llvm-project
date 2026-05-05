add_mlir_dialect(WaveMachineOps wavemachine)
add_mlir_doc(WaveMachineOps WaveMachineOps Dialects/ -gen-dialect-doc)

set(LLVM_TARGET_DEFINITIONS WaveMachineOps.td)
mlir_tablegen(WaveMachineOpsEnums.h.inc -gen-enum-decls)
mlir_tablegen(WaveMachineOpsEnums.cpp.inc -gen-enum-defs)
add_public_tablegen_target(MLIRWaveMachineEnumsIncGen)

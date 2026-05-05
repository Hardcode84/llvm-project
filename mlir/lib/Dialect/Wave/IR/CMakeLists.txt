add_mlir_dialect_library(MLIRWaveDialect
  PARTIAL_SOURCES_INTENDED
  Wave.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/Wave

  DEPENDS
  MLIRWaveOpsIncGen

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRGPUDialect
  MLIRIR
  MLIRMemRefDialect
  MLIRSCFDialect
  MLIRSideEffectInterfaces
  )

add_mlir_dialect_library(MLIRWaveAMDDialect
  PARTIAL_SOURCES_INTENDED
  WaveAMD.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/Wave

  DEPENDS
  MLIRWaveAMDOpsIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRMemRefDialect
  MLIRSideEffectInterfaces
  MLIRWaveDialect
  )

add_mlir_dialect_library(MLIRWaveTransforms
  WaveToROCDL.cpp
  WaveToGPU.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/Wave

  DEPENDS
  MLIRWaveTransformsIncGen

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRGPUDialect
  MLIRIR
  MLIRROCDLDialect
  MLIRPass
  MLIRSCFDialect
  MLIRTransformUtils
  MLIRWaveDialect
  )

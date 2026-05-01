add_mlir_dialect_library(MLIRWaveTransforms
  WaveMachine.cpp
  WaveToROCDL.cpp
  WaveToGPU.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/Wave

  DEPENDS
  MLIRWaveTransformsIncGen

  LINK_COMPONENTS
  AMDGPUUtils
  TargetParser

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRGPUDialect
  MLIRIR
  MLIRMemRefDialect
  MLIRROCDLDialect
  MLIRPass
  MLIRSCFDialect
  MLIRTransformUtils
  MLIRWaveDialect
  MLIRWaveMachineDialect
  )

target_include_directories(obj.MLIRWaveTransforms
  PRIVATE
  ${LLVM_MAIN_SRC_DIR}/lib/Target/AMDGPU
  ${LLVM_BINARY_DIR}/lib/Target/AMDGPU
  )

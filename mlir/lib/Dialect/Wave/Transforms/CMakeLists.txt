add_mlir_dialect_library(MLIRWaveTransforms
  WaveAMDABILowering.cpp
  WaveAMDHazardWaits.cpp
  WaveAMDMetadata.cpp
  WaveAMDRegAlloc.cpp
  WaveAMDResourceInfo.cpp
  WaveMachine.cpp
  WaveMachineWaitcnt.cpp
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
  MLIRAnalysis
  MLIRArithDialect
  MLIRGPUDialect
  MLIRIR
  MLIRROCDLDialect
  MLIRPass
  MLIRSCFDialect
  MLIRTransformUtils
  MLIRWaveAMDDialect
  MLIRWaveDialect
  MLIRWaveMachineDialect
  )

target_include_directories(obj.MLIRWaveTransforms
  PRIVATE
  ${LLVM_MAIN_SRC_DIR}/lib/Target/AMDGPU
  ${LLVM_BINARY_DIR}/lib/Target/AMDGPU
  )

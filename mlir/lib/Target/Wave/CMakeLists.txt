add_mlir_translation_library(MLIRTargetWave
  AMDGPU.cpp
  TranslateRegistration.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Target/Wave

  LINK_COMPONENTS
  AMDGPUAsmParser
  AMDGPUDesc
  AMDGPUInfo
  AMDGPUUtils
  MC
  Support
  TargetParser

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRFuncDialect
  MLIRIR
  MLIRSupport
  MLIRTranslateLib
  MLIRWaveDialect
  MLIRWaveMachineDialect
  MLIRWaveTransforms
  )

target_include_directories(obj.MLIRTargetWave
  PRIVATE
  ${LLVM_MAIN_SRC_DIR}/lib/Target/AMDGPU
  ${LLVM_BINARY_DIR}/lib/Target/AMDGPU
  )

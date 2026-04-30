add_mlir_translation_library(MLIRTargetWave
  AMDGPU.cpp
  TranslateRegistration.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Target/Wave

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRFuncDialect
  MLIRIR
  MLIRSupport
  MLIRTranslateLib
  MLIRWaveDialect
  )

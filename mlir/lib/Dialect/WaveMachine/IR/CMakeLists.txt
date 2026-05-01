add_mlir_dialect_library(MLIRWaveMachineDialect
  WaveMachine.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/WaveMachine

  DEPENDS
  MLIRWaveMachineOpsIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  )

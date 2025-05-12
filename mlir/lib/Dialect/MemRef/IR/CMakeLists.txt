add_mlir_dialect_library(MLIRMemRefDialect
  MemRefDialect.cpp
  MemoryAccessOpInterfaces.cpp
  MemRefMemorySlot.cpp
  MemRefOps.cpp
  ValueBoundsOpInterfaceImpl.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/inlude/mlir/Dialect/MemRefDialect

  DEPENDS
  MLIRMemRefOpsIncGen
  MLIRMemoryAccessOpInterfacesIncGen

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRArithUtils
  MLIRControlFlowInterfaces
  MLIRDialect
  MLIRDialectUtils
  MLIRInferIntRangeCommon
  MLIRInferIntRangeInterface
  MLIRInferTypeOpInterface
  MLIRIR
  MLIRMemOpInterfaces
  MLIRMemorySlotInterfaces
  MLIRShapedOpInterfaces
  MLIRSideEffectInterfaces
  MLIRValueBoundsOpInterface
  MLIRViewLikeInterface
)

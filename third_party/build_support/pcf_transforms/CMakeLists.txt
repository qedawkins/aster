# Build LowerStructuralPCF pass from IREE submodule sources.
#
# TableGen outputs land in CMAKE_CURRENT_BINARY_DIR which is set to
# <build>/third_party/gen/iree/compiler/Codegen/Dialect/PCF/Transforms/
# by the parent add_subdirectory() call.

set(IREE_PCF_TRANSFORMS_SRC
  ${IREE_COMPILER_SRC_ROOT}/iree/compiler/Codegen/Dialect/PCF/Transforms)

# Use our subset Passes.td (only LowerStructuralPCFPass).
set(LLVM_TARGET_DEFINITIONS
  ${CMAKE_CURRENT_SOURCE_DIR}/Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name=PCF)
add_public_tablegen_target(PCFPassesIncGen)

# Build the PCF transforms library.
# LowerStructuralPCF.cpp comes from IREE source, Passes.cpp is ours.
add_mlir_library(PCFTransforms
  ${IREE_PCF_TRANSFORMS_SRC}/LowerStructuralPCF.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/Passes.cpp

  DEPENDS
  PCFPassesIncGen

  LINK_LIBS PUBLIC
  PCFDialect
  LLVMSupport
  MLIRControlFlowDialect
  MLIRIR
  MLIRPass
  MLIRSCFDialect
  MLIRSupport
  MLIRTransformUtils
)

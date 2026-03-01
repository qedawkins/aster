# Build PCF IR dialect from IREE submodule sources.
#
# TableGen outputs land in CMAKE_CURRENT_BINARY_DIR which is set to
# <build>/third_party/gen/iree/compiler/Codegen/Dialect/PCF/IR/
# by the parent add_subdirectory() call.

set(IREE_PCF_IR_SRC
  ${IREE_COMPILER_SRC_ROOT}/iree/compiler/Codegen/Dialect/PCF/IR)

# TableGen needs to find iree/compiler/... includes in .td files.
list(APPEND LLVM_TABLEGEN_FLAGS "-I${IREE_COMPILER_SRC_ROOT}")

# --- Interfaces ---
set(LLVM_TARGET_DEFINITIONS ${IREE_PCF_IR_SRC}/PCFInterfaces.td)
mlir_tablegen(PCFAttrInterfaces.h.inc -gen-attr-interface-decls)
mlir_tablegen(PCFAttrInterfaces.cpp.inc -gen-attr-interface-defs)
mlir_tablegen(PCFOpInterfaces.h.inc -gen-op-interface-decls)
mlir_tablegen(PCFOpInterfaces.cpp.inc -gen-op-interface-defs)
add_public_tablegen_target(PCFInterfacesIncGen)

# --- Ops ---
set(LLVM_TARGET_DEFINITIONS ${IREE_PCF_IR_SRC}/PCFOps.td)
mlir_tablegen(PCFOps.h.inc -gen-op-decls)
mlir_tablegen(PCFOps.cpp.inc -gen-op-defs)
add_public_tablegen_target(PCFOpsIncGen)

# --- Attrs and Types (from PCFBase.td) ---
set(LLVM_TARGET_DEFINITIONS ${IREE_PCF_IR_SRC}/PCFBase.td)
mlir_tablegen(PCFAttrs.h.inc -gen-attrdef-decls -attrdefs-dialect=pcf)
mlir_tablegen(PCFAttrs.cpp.inc -gen-attrdef-defs -attrdefs-dialect=pcf)
mlir_tablegen(PCFTypes.h.inc -gen-typedef-decls -typedefs-dialect=pcf)
mlir_tablegen(PCFTypes.cpp.inc -gen-typedef-defs -typedefs-dialect=pcf)
add_public_tablegen_target(PCFTypesIncGen)

# --- Dialect library ---
add_mlir_dialect_library(PCFDialect
  ${IREE_PCF_IR_SRC}/PCFAttrs.cpp
  ${IREE_PCF_IR_SRC}/PCFDialect.cpp
  ${IREE_PCF_IR_SRC}/PCFInterfaces.cpp
  ${IREE_PCF_IR_SRC}/PCFOps.cpp
  ${IREE_PCF_IR_SRC}/PCFTypes.cpp

  DEPENDS
  PCFInterfacesIncGen
  PCFOpsIncGen
  PCFTypesIncGen

  LINK_LIBS PUBLIC
  LLVMSupport
  MLIRArithDialect
  MLIRControlFlowInterfaces
  MLIRIR
  MLIRInferTypeOpInterface
  MLIRParser
  MLIRSideEffectInterfaces
  MLIRSupport
  MLIRTransformUtils
  MLIRViewLikeInterface
)

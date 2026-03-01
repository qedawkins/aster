add_mlir_dialect_library(AMDGCNDialect
  AMDGCN.cpp
  AMDGCNAttrs.cpp
  AMDGCNScopeModels.cpp
  AMDGCNTypes.cpp
  AMDGCNVerifiers.cpp
  Utils.cpp

  DEPENDS
  AMDGCNAttrInterfacesIncGen
  AMDGCNInstOpInterfaceIncGen
  AMDGCNInstsIncGen
  AMDGCNOpsIncGen
  AMDGCNTypeInterfacesIncGen
  PCFInterfacesIncGen

  LINK_LIBS PUBLIC
  ASTERIR
  AsterInterfaces
  LSIRDialect
  PCFDialect
  MLIRArithDialect
  MLIRControlFlowDialect
  MLIRControlFlowInterfaces
  MLIRFunctionInterfaces
  MLIRGPUDialect
  MLIRIR
  MLIRInferTypeOpInterface
  MLIRPtrDialect
)

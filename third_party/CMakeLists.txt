# Build PCF dialect from IREE submodule sources.
#
# We compile ONLY the PCF IR dialect and the LowerStructuralPCF pass.
# No other IREE components are built. IREE's LLVM/MLIR dependency is
# satisfied by aster's shared LLVM installation.
#
# Include paths for iree/compiler/... are set in the top-level CMakeLists.txt:
#   - Source headers: ${IREE_COMPILER_SRC_ROOT} (third_party/iree/compiler/src/)
#   - Generated .inc: ${IREE_PCF_GEN_ROOT} (<build>/third_party/gen/)

# Generated .inc files must land at paths matching the IREE include structure.
# We use explicit binary dirs in add_subdirectory() so that cmake puts generated
# files under <build>/third_party/gen/iree/compiler/Codegen/Dialect/PCF/...
add_subdirectory(build_support/pcf_ir
  ${IREE_PCF_GEN_ROOT}/iree/compiler/Codegen/Dialect/PCF/IR)
add_subdirectory(build_support/pcf_transforms
  ${IREE_PCF_GEN_ROOT}/iree/compiler/Codegen/Dialect/PCF/Transforms)

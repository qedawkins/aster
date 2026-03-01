// Aster-specific PCF pass registration.
//
// Only registers the subset of PCF passes that we build from IREE sources.

#include "mlir/Pass/Pass.h"

// Pull in the generated pass declarations and registration from our
// subset Passes.td (only LowerStructuralPCFPass).
namespace mlir::iree_compiler::IREE::PCF {
#define GEN_PASS_DECL
#include "iree/compiler/Codegen/Dialect/PCF/Transforms/Passes.h.inc"
#define GEN_PASS_REGISTRATION
#include "iree/compiler/Codegen/Dialect/PCF/Transforms/Passes.h.inc"
} // namespace mlir::iree_compiler::IREE::PCF

namespace mlir::aster {
void registerPCFPasses() {
  iree_compiler::IREE::PCF::registerLowerStructuralPCFPass();
}
} // namespace mlir::aster

// Aster-specific Passes.td: subset of PCF passes built from IREE sources.
//
// Only LowerStructuralPCFPass is needed. The other PCF passes
// (FuseConsumers, ConvertSRefToMemRef, etc.) have IREE-specific deps.

#ifndef ASTER_THIRD_PARTY_PCF_TRANSFORMS_PASSES
#define ASTER_THIRD_PARTY_PCF_TRANSFORMS_PASSES

include "mlir/Pass/PassBase.td"

def LowerStructuralPCFPass : Pass<"pcf-lower-structural-pcf", ""> {
  let summary = "Lowers pcf.generic/loop to scf.execute_region/forall";
  let description = [{
    Lowers structured PCF parallel ops to SCF and CF ops. The scope attribute
    on each op determines how worker IDs and counts are materialized.

    Op conversions:
    - `pcf.generic` -> `scf.execute_region` with worker IDs from scope
    - `pcf.loop` -> serialized `scf.forall` with iteration bounds from count
    - `pcf.return` -> `scf.yield` (in generic) or `scf.forall.in_parallel`
    - `pcf.branch_cond_return` -> `cf.cond_br` to a return block
  }];
  let dependentDialects = ["::mlir::iree_compiler::IREE::PCF::PCFDialect",
                           "::mlir::scf::SCFDialect",
                           "::mlir::cf::ControlFlowDialect"];
}

#endif // ASTER_THIRD_PARTY_PCF_TRANSFORMS_PASSES

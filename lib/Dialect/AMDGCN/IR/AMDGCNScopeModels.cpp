//===- AMDGCNScopeModels.cpp - PCF scope models for AMDGCN ----------------===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implements ScopeAttrInterface for ThreadScopeAttr and registers the
// PCFConversionDialectInterface for the AMDGCN dialect.
//
//===----------------------------------------------------------------------===//

#include "aster/Dialect/AMDGCN/IR/AMDGCNScopeModels.h"

#include "aster/Dialect/AMDGCN/IR/AMDGCNAttrs.h"
#include "aster/Dialect/AMDGCN/IR/AMDGCNDialect.h"
#include "iree/compiler/Codegen/Dialect/PCF/IR/PCFInterfaces.h"
#include "iree/compiler/Codegen/Dialect/PCF/Transforms/ConversionDialectInterface.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/IR/Builders.h"

namespace iree_pcf = mlir::iree_compiler::IREE::PCF;

namespace mlir::aster::amdgcn {

namespace {

/// Number of lanes per subgroup on AMDGCN targets.
static constexpr int64_t kLanesPerSubgroup = 64;

/// Dialect interface implementation for AMDGCN that loads dependent dialects
/// needed during PCF structural lowering (which generates GPU/arith ops).
class AMDGCNConversionDialectInterface final
    : public iree_compiler::PCFConversionDialectInterface {
public:
  AMDGCNConversionDialectInterface(Dialect *dialect)
      : iree_compiler::PCFConversionDialectInterface(dialect) {}

  void
  loadStructuralLoweringDependentDialects(MLIRContext *context) const override {
    context->loadDialect<gpu::GPUDialect, arith::ArithDialect>();
  }

  void loadSRefLoweringDependentDialects(MLIRContext *context) const override {}
};

/// External model implementing ScopeAttrInterface for ThreadScopeAttr.
///
/// Without a subgroup grid, provides two levels of thread IDs:
///   - lane_id:     gpu.thread_id x mod 64
///   - subgroup_id: gpu.thread_id x floordiv 64
///
/// With a subgroup grid (e.g., [2, 2]), the subgroup_id is delinearized
/// into multiple dimensions following affine.delinearize_index semantics
/// (outermost dimension first):
///   - lane_id:   gpu.thread_id x mod 64
///   - grid_id_0: subgroup_id / product(grid[1:])   (outermost)
///   - grid_id_1: (subgroup_id / product(grid[2:])) % grid[1]
///   - ...
///   - grid_id_N: subgroup_id % grid[N]             (innermost)
///
/// When a single ID is requested, the flat gpu.thread_id x is returned.
struct ThreadScopeModel
    : public iree_pcf::ScopeAttrInterface::ExternalModel<ThreadScopeModel,
                                                         ThreadScopeAttr> {
  SmallVector<Value> getWorkerCounts(Attribute attr, OpBuilder &builder,
                                     Location loc, int64_t numIds) const {
    assert(numIds >= 1 && "expected at least one requested worker count");
    ArrayRef<int64_t> grid = cast<ThreadScopeAttr>(attr).getSubgroupGrid();
    SmallVector<Value> counts(numIds, Value());

    if (numIds == 1) {
      // Single ID: total thread count = gpu.block_dim x.
      counts.front() = gpu::BlockDimOp::create(builder, loc, gpu::Dimension::x);
      return counts;
    }

    // Lane count is always 64.
    counts[0] = arith::ConstantIndexOp::create(builder, loc, kLanesPerSubgroup);

    if (!grid.empty()) {
      // Grid-based counts: each grid dimension becomes a static count.
      int64_t gridSize = static_cast<int64_t>(grid.size());
      int64_t numGridIds = std::min(numIds - 1, gridSize);
      for (int64_t i = 0; i < numGridIds; ++i) {
        counts[1 + i] = arith::ConstantIndexOp::create(builder, loc, grid[i]);
      }
      // Pad remaining counts with 1.
      if (1 + numGridIds < numIds) {
        Value one = arith::ConstantIndexOp::create(builder, loc, 1);
        for (int64_t i = 1 + numGridIds; i < numIds; ++i) {
          counts[i] = one;
        }
      }
    } else {
      // No grid: subgroup_count = block_dim_x / 64.
      Value blockDim = gpu::BlockDimOp::create(builder, loc, gpu::Dimension::x);
      counts[1] = arith::DivUIOp::create(builder, loc, blockDim, counts[0]);
      // Pad remaining counts with 1.
      if (numIds > 2) {
        Value one = arith::ConstantIndexOp::create(builder, loc, 1);
        for (int64_t i = 2; i < numIds; ++i) {
          counts[i] = one;
        }
      }
    }

    return counts;
  }

  SmallVector<Value> getWorkerIDs(Attribute attr, OpBuilder &builder,
                                  Location loc, int64_t numIds) const {
    assert(numIds >= 1 && "expected at least one requested worker id");
    ArrayRef<int64_t> grid = cast<ThreadScopeAttr>(attr).getSubgroupGrid();
    SmallVector<Value> ids(numIds, Value());

    Value threadId = gpu::ThreadIdOp::create(builder, loc, gpu::Dimension::x);

    if (numIds == 1) {
      // Single ID: flat thread id.
      ids.front() = threadId;
      return ids;
    }

    // lane_id = thread_id % 64.
    Value lanesPerSubgroup =
        arith::ConstantIndexOp::create(builder, loc, kLanesPerSubgroup);
    ids[0] = arith::RemUIOp::create(builder, loc, threadId, lanesPerSubgroup);

    // subgroup_id = thread_id / 64.
    Value subgroupId =
        arith::DivUIOp::create(builder, loc, threadId, lanesPerSubgroup);

    if (!grid.empty()) {
      // Delinearize subgroup_id by the grid (outermost-first convention).
      // For grid = [D0, D1, ..., DN]:
      //   id_0 = subgroup_id / (D1 * D2 * ... * DN)
      //   id_1 = (subgroup_id / (D2 * ... * DN)) % D1
      //   ...
      //   id_N = subgroup_id % DN
      int64_t gridSize = static_cast<int64_t>(grid.size());
      int64_t numGridIds = std::min(numIds - 1, gridSize);

      // Compute suffix products: suffixProd[i] = grid[i+1] * ... * grid[N-1].
      SmallVector<int64_t> suffixProd(gridSize, 1);
      for (int64_t i = gridSize - 2; i >= 0; --i) {
        suffixProd[i] = suffixProd[i + 1] * grid[i + 1];
      }

      for (int64_t i = 0; i < numGridIds; ++i) {
        Value current = subgroupId;
        // Divide by suffix product to get the quotient for this dim.
        if (suffixProd[i] > 1) {
          Value divisor =
              arith::ConstantIndexOp::create(builder, loc, suffixProd[i]);
          current = arith::DivUIOp::create(builder, loc, current, divisor);
        }
        // Take modulo of the grid dimension (except for the outermost dim).
        if (i > 0) {
          Value modulus = arith::ConstantIndexOp::create(builder, loc, grid[i]);
          current = arith::RemUIOp::create(builder, loc, current, modulus);
        }
        ids[1 + i] = current;
      }

      // Pad remaining ids with 0.
      if (numIds > 1 + numGridIds) {
        Value zero = arith::ConstantIndexOp::create(builder, loc, 0);
        for (int64_t i = 1 + numGridIds; i < numIds; ++i) {
          ids[i] = zero;
        }
      }
    } else {
      // No grid: single subgroup_id.
      ids[1] = subgroupId;
      // Pad remaining ids with 0.
      if (numIds > 2) {
        Value zero = arith::ConstantIndexOp::create(builder, loc, 0);
        for (int64_t i = 2; i < numIds; ++i) {
          ids[i] = zero;
        }
      }
    }

    return ids;
  }

  LogicalResult addBarrier(Attribute attr, OpBuilder &builder) const {
    gpu::BarrierOp::create(builder, builder.getUnknownLoc());
    return success();
  }

  FailureOr<Attribute> getAllocMemSpace(Attribute attr,
                                        MLIRContext *context) const {
    // Thread scope allocations target workgroup-shared memory.
    return gpu::AddressSpaceAttr::get(context, gpu::AddressSpace::Workgroup);
  }
};

} // namespace

void registerAMDGCNScopeExternalModels(DialectRegistry &registry) {
  registry.addExtension(
      +[](MLIRContext *context, amdgcn::AMDGCNDialect *dialect) {
        ThreadScopeAttr::attachInterface<ThreadScopeModel>(*context);
        dialect->addInterfaces<AMDGCNConversionDialectInterface>();
      });
}

} // namespace mlir::aster::amdgcn

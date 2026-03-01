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
/// Provides two levels of thread IDs (fastest to slowest varying):
///   - lane_id:     gpu.thread_id x mod 64
///   - subgroup_id: gpu.thread_id x floordiv 64
///
/// And corresponding counts:
///   - lane_count:     64 (constant)
///   - subgroup_count: gpu.block_dim x / 64
///
/// When a single ID is requested, the flat gpu.thread_id x is returned.
struct ThreadScopeModel
    : public iree_pcf::ScopeAttrInterface::ExternalModel<ThreadScopeModel,
                                                         ThreadScopeAttr> {
  SmallVector<Value> getWorkerCounts(Attribute attr, OpBuilder &builder,
                                     Location loc, int64_t numIds) const {
    assert(numIds >= 1 && "expected at least one requested worker count");
    SmallVector<Value> counts(numIds, Value());

    if (numIds == 1) {
      // Single ID: total thread count = gpu.block_dim x.
      counts.front() =
          gpu::BlockDimOp::create(builder, loc, gpu::Dimension::x);
    } else {
      // Two or more IDs: fastest = 64 (lanes), next = block_dim_x / 64.
      Value laneCount =
          arith::ConstantIndexOp::create(builder, loc, kLanesPerSubgroup);
      counts[0] = laneCount;

      Value blockDim =
          gpu::BlockDimOp::create(builder, loc, gpu::Dimension::x);
      counts[1] = arith::DivUIOp::create(builder, loc, blockDim, laneCount);

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
    SmallVector<Value> ids(numIds, Value());

    Value threadId =
        gpu::ThreadIdOp::create(builder, loc, gpu::Dimension::x);

    if (numIds == 1) {
      // Single ID: flat thread id.
      ids.front() = threadId;
    } else {
      // Two or more IDs: fastest = lane_id, next = subgroup_id.
      Value lanesPerSubgroup =
          arith::ConstantIndexOp::create(builder, loc, kLanesPerSubgroup);
      // lane_id = thread_id % 64.
      ids[0] = arith::RemUIOp::create(builder, loc, threadId, lanesPerSubgroup);
      // subgroup_id = thread_id / 64.
      ids[1] =
          arith::DivUIOp::create(builder, loc, threadId, lanesPerSubgroup);

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

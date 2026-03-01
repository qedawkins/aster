//===- AMDGCNScopeModels.h - PCF scope models for AMDGCN --------*- C++ -*-===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Registers ScopeAttrInterface external models for AMDGCN dialect attributes.
//
//===----------------------------------------------------------------------===//

#ifndef ASTER_DIALECT_AMDGCN_IR_AMDGCNSCOPEMODELS_H
#define ASTER_DIALECT_AMDGCN_IR_AMDGCNSCOPEMODELS_H

namespace mlir {
class DialectRegistry;
} // namespace mlir

namespace mlir::aster::amdgcn {

/// Registers ScopeAttrInterface external models for AMDGCN attributes
/// (ThreadScopeAttr) and the PCFConversionDialectInterface for the AMDGCN
/// dialect.
void registerAMDGCNScopeExternalModels(DialectRegistry &registry);

} // namespace mlir::aster::amdgcn

#endif // ASTER_DIALECT_AMDGCN_IR_AMDGCNSCOPEMODELS_H

//===- WaveMachineTraits.h - WaveMachine op traits -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_WAVEMACHINE_IR_WAVEMACHINETRAITS_H
#define MLIR_DIALECT_WAVEMACHINE_IR_WAVEMACHINETRAITS_H

#include "mlir/IR/OpDefinition.h"

namespace mlir::OpTrait::wavemachine {

template <typename ConcreteType>
class VALUOp : public TraitBase<ConcreteType, VALUOp> {};

template <typename ConcreteType>
class SMEMLoadOp : public TraitBase<ConcreteType, SMEMLoadOp> {};

template <typename ConcreteType>
class VMEMLoadOp : public TraitBase<ConcreteType, VMEMLoadOp> {};

template <typename ConcreteType>
class VMEMStoreOp : public TraitBase<ConcreteType, VMEMStoreOp> {};

template <typename ConcreteType>
class WaitcntOp : public TraitBase<ConcreteType, WaitcntOp> {};

template <typename ConcreteType>
class TokenOp : public TraitBase<ConcreteType, TokenOp> {};

template <typename ConcreteType>
class TokenJoinOp : public TraitBase<ConcreteType, TokenJoinOp> {};

} // namespace mlir::OpTrait::wavemachine

#endif // MLIR_DIALECT_WAVEMACHINE_IR_WAVEMACHINETRAITS_H

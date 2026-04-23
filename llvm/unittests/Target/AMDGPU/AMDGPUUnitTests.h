//===---------- llvm/unittests/Target/AMDGPU/AMDGPUUnitTests.h ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_UNITTESTS_TARGET_AMDGPU_AMDGPUUNITTESTS_H
#define LLVM_UNITTESTS_TARGET_AMDGPU_AMDGPUUNITTESTS_H

#include <memory>
#include <string>
#include "llvm/Support/Compiler.h"

LLVM_NAMESPACE_BEGIN

class GCNTargetMachine;
class StringRef;

std::unique_ptr<const GCNTargetMachine>
createAMDGPUTargetMachine(std::string TStr, StringRef CPU, StringRef FS);

LLVM_NAMESPACE_END // end namespace llvm

#endif // LLVM_UNITTESTS_TARGET_AMDGPU_AMDGPUUNITTESTS_H

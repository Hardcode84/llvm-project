//===-- AVRTargetStreamer.cpp - AVR Target Streamer Methods ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file provides AVR specific target streamer methods.
//
//===----------------------------------------------------------------------===//

#include "AVRTargetStreamer.h"

#include "llvm/MC/MCContext.h"
#include "llvm/Support/Compiler.h"

LLVM_NAMESPACE_BEGIN

AVRTargetStreamer::AVRTargetStreamer(MCStreamer &S) : MCTargetStreamer(S) {}

AVRTargetAsmStreamer::AVRTargetAsmStreamer(MCStreamer &S)
    : AVRTargetStreamer(S) {}

LLVM_NAMESPACE_END // end namespace llvm

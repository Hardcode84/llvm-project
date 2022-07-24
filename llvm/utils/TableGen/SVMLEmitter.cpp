//===------ SVMLEmitter.cpp - Generate SVML function variants -------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This tablegen backend emits the scalar to svml function map for TLI.
//
//===----------------------------------------------------------------------===//

#include "CodeGenTarget.h"
#include "llvm/Support/Format.h"
#include "llvm/TableGen/Error.h"
#include "llvm/TableGen/Record.h"
#include "llvm/TableGen/TableGenBackend.h"
#include <map>
#include <vector>

using namespace llvm;

#define DEBUG_TYPE "SVMLVariants"
#include "llvm/Support/Debug.h"

namespace {

class SVMLVariantsEmitter {

  RecordKeeper &Records;

private:
  void emitSVMLVariants(raw_ostream &OS);

public:
  SVMLVariantsEmitter(RecordKeeper &R) : Records(R) {}

  void run(raw_ostream &OS);
};
} // End anonymous namespace

/// \brief Emit the set of SVML variant function names.
// The default is to emit the high accuracy SVML variants until a mechanism is
// introduced to allow a selection of different variants through precision
// requirements specified by the user. This code generates mappings to svml
// that are in the scalar form of llvm intrinsics, math library calls, or the
// finite variants of math library calls.
void SVMLVariantsEmitter::emitSVMLVariants(raw_ostream &OS) {

  const unsigned MinSinglePrecVL = 4;
  const unsigned MaxSinglePrecVL = 16;
  const unsigned MinDoublePrecVL = 2;
  const unsigned MaxDoublePrecVL = 8;

  OS << "#ifdef GET_SVML_VARIANTS\n";

  for (const auto &D : Records.getAllDerivedDefinitions("SvmlVariant")) {
    StringRef SvmlVariantNameStr = D->getName();
    // Single Precision SVML
    for (unsigned VL = MinSinglePrecVL; VL <= MaxSinglePrecVL; VL *= 2) {
      // Emit the scalar math library function to svml function entry.
      OS << "{\"" << SvmlVariantNameStr << "f" << "\", ";
      OS << "\"" << "__svml_" << SvmlVariantNameStr << "f" << VL << "\", "
         << "ElementCount::getFixed(" << VL << ")},\n";

      // Emit the scalar intrinsic to svml function entry.
      OS << "{\"" << "llvm." << SvmlVariantNameStr << ".f32" << "\", ";
      OS << "\"" << "__svml_" << SvmlVariantNameStr << "f" << VL << "\", "
         << "ElementCount::getFixed(" << VL << ")},\n";

      // Emit the finite math library function to svml function entry.
      OS << "{\"__" << SvmlVariantNameStr << "f_finite" << "\", ";
      OS << "\"" << "__svml_" << SvmlVariantNameStr << "f" << VL << "\", "
         << "ElementCount::getFixed(" << VL << ")},\n";
    }

    // Double Precision SVML
    for (unsigned VL = MinDoublePrecVL; VL <= MaxDoublePrecVL; VL *= 2) {
      // Emit the scalar math library function to svml function entry.
      OS << "{\"" << SvmlVariantNameStr << "\", ";
      OS << "\"" << "__svml_" << SvmlVariantNameStr << VL << "\", " << "ElementCount::getFixed(" << VL
         << ")},\n";

      // Emit the scalar intrinsic to svml function entry.
      OS << "{\"" << "llvm." << SvmlVariantNameStr << ".f64" << "\", ";
      OS << "\"" << "__svml_" << SvmlVariantNameStr << VL << "\", " << "ElementCount::getFixed(" << VL
         << ")},\n";

      // Emit the finite math library function to svml function entry.
      OS << "{\"__" << SvmlVariantNameStr << "_finite" << "\", ";
      OS << "\"" << "__svml_" << SvmlVariantNameStr << VL << "\", "
         << "ElementCount::getFixed(" << VL << ")},\n";
    }
  }

  OS << "#endif // GET_SVML_VARIANTS\n\n";
}

void SVMLVariantsEmitter::run(raw_ostream &OS) {
  emitSVMLVariants(OS);
}

namespace llvm {

void EmitSVMLVariants(RecordKeeper &RK, raw_ostream &OS) {
  SVMLVariantsEmitter(RK).run(OS);
}

} // End llvm namespace

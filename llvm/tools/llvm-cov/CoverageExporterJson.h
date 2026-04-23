//===- CoverageExporterJson.h - Code coverage JSON exporter ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This class implements a code coverage exporter for JSON format.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_COV_COVERAGEEXPORTERJSON_H
#define LLVM_COV_COVERAGEEXPORTERJSON_H

#include "CoverageExporter.h"
#include "llvm/Support/Compiler.h"

LLVM_NAMESPACE_BEGIN

class CoverageExporterJson : public CoverageExporter {
public:
  CoverageExporterJson(const coverage::CoverageMapping &CoverageMapping,
                       const CoverageViewOptions &Options, raw_ostream &OS)
      : CoverageExporter(CoverageMapping, Options, OS) {}

  /// Render the CoverageMapping object.
  void renderRoot(const CoverageFilters &IgnoreFilters) override;

  /// Render the CoverageMapping object for specified source files.
  void renderRoot(ArrayRef<std::string> SourceFiles) override;
};

LLVM_NAMESPACE_END // end namespace llvm

#endif // LLVM_COV_COVERAGEEXPORTERJSON_H

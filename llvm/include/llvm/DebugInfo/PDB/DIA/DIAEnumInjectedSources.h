//==- DIAEnumInjectedSources.h - DIA Injected Sources Enumerator -*- C++ -*-==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_DEBUGINFO_PDB_DIA_DIAENUMINJECTEDSOURCES_H
#define LLVM_DEBUGINFO_PDB_DIA_DIAENUMINJECTEDSOURCES_H

#include "DIASupport.h"
#include "llvm/DebugInfo/PDB/IPDBEnumChildren.h"
#include "llvm/DebugInfo/PDB/IPDBInjectedSource.h"
#include "llvm/Support/Compiler.h"

LLVM_NAMESPACE_BEGIN
namespace pdb {

class DIAEnumInjectedSources : public IPDBEnumChildren<IPDBInjectedSource> {
public:
  explicit DIAEnumInjectedSources(
      CComPtr<IDiaEnumInjectedSources> DiaEnumerator);

  uint32_t getChildCount() const override;
  ChildTypePtr getChildAtIndex(uint32_t Index) const override;
  ChildTypePtr getNext() override;
  void reset() override;

private:
  CComPtr<IDiaEnumInjectedSources> Enumerator;
};
} // namespace pdb
LLVM_NAMESPACE_END // namespace llvm

#endif // LLVM_DEBUGINFO_PDB_DIA_DIAENUMINJECTEDSOURCES_H

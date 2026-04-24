//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines common utilities for generating C++ code.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TABLEGEN_CODEGENHELPERS_H
#define LLVM_TABLEGEN_CODEGENHELPERS_H

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"
#include <string>
#include "llvm/Support/Compiler.h"

LLVM_NAMESPACE_BEGIN

// Simple RAII helper for emitting ifdef-undef-endif scope. `LateUndef` controls
// whether the undef is emitted at the start of the scope (false) or at the end
// of the scope (true).
class IfDefEmitter {
public:
  IfDefEmitter(raw_ostream &OS, StringRef Name, bool LateUndef = false)
      : Name(Name.str()), OS(OS), LateUndef(LateUndef) {
    OS << "#ifdef " << Name << "\n";
    if (!LateUndef)
      OS << "#undef " << Name << "\n";
    OS << "\n";
  }

  IfDefEmitter(const IfDefEmitter &) = delete;
  IfDefEmitter &operator=(const IfDefEmitter &) = delete;

  ~IfDefEmitter() {
    OS << "\n";
    if (LateUndef)
      OS << "#undef " << Name << "\n";
    OS << "#endif // " << Name << "\n\n";
  }

private:
  std::string Name;
  raw_ostream &OS;
  bool LateUndef;
};

// Base class for RAII helpers for emitting various if guards.
class IfGuardEmitterBase {
protected:
  IfGuardEmitterBase(raw_ostream &OS, StringRef If, StringRef Condition)
      : Condition(Condition.str()), OS(OS) {
    OS << If << " " << Condition << "\n\n";
  }

  IfGuardEmitterBase(const IfGuardEmitterBase &) = delete;
  IfGuardEmitterBase &operator=(const IfGuardEmitterBase &) = delete;

  ~IfGuardEmitterBase() { OS << "\n#endif // " << Condition << "\n\n"; }

private:
  std::string Condition;
  raw_ostream &OS;
};

// RAII emitter for:
// #if <Condition>
// #endif // <Condition>
class IfGuardEmitter : private IfGuardEmitterBase {
public:
  IfGuardEmitter(raw_ostream &OS, StringRef Condition)
      : IfGuardEmitterBase(OS, "#if", Condition) {}
};

// RAII emitter for:
// #ifdef <Condition>
// #endif // <Condition>
class IfDefGuardEmitter : private IfGuardEmitterBase {
public:
  IfDefGuardEmitter(raw_ostream &OS, StringRef Condition)
      : IfGuardEmitterBase(OS, "#ifdef", Condition) {}
};

// RAII emitter for:
// #ifndef <Condition>
// #endif // <Condition>
class IfNDefGuardEmitter : private IfGuardEmitterBase {
public:
  IfNDefGuardEmitter(raw_ostream &OS, StringRef Condition)
      : IfGuardEmitterBase(OS, "#ifndef", Condition) {}
};

// Simple RAII helper for emitting header include guard (ifndef-define-endif).
class IncludeGuardEmitter {
public:
  IncludeGuardEmitter(raw_ostream &OS, StringRef Name)
      : Name(Name.str()), OS(OS) {
    OS << "#ifndef " << Name << "\n"
       << "#define " << Name << "\n\n";
  }

  IncludeGuardEmitter(const IncludeGuardEmitter &) = delete;
  IncludeGuardEmitter &operator=(const IncludeGuardEmitter &) = delete;

  ~IncludeGuardEmitter() { OS << "\n#endif // " << Name << "\n\n"; }

private:
  std::string Name;
  raw_ostream &OS;
};

// Simple RAII helper for emitting namespace scope. Name can be a single
// namespace or nested namespace. If the name is empty, will not generate any
// namespace scope.
//
// Names that start with ``llvm`` (standalone or as a nested prefix like
// ``llvm::foo``) or ``mlir`` are emitted via the ``LLVM_NAMESPACE_BEGIN``
// or ``MLIR_NAMESPACE_BEGIN`` transport macros so the inline ABI
// namespace tag applies uniformly to TableGen-generated ``.inc`` files.
// Any inner nested namespace (e.g. ``foo`` in ``llvm::foo``) is emitted
// as a plain ``namespace foo { ... }`` inside the macro expansion.
class NamespaceEmitter {
public:
  NamespaceEmitter(raw_ostream &OS, const Twine &NameUntrimmed)
      : Name(trim(NameUntrimmed.str()).str()), OS(OS) {
    if (Name.empty())
      return;
    StringRef First, Rest;
    std::tie(First, Rest) = StringRef(Name).split("::");
    if (First == "llvm") {
      OS << "LLVM_NAMESPACE_BEGIN\n";
      MacroKind = Kind::LLVM;
      if (!Rest.empty())
        OS << "namespace " << Rest << " {\n";
    } else if (First == "mlir") {
      OS << "MLIR_NAMESPACE_BEGIN\n";
      MacroKind = Kind::MLIR;
      if (!Rest.empty())
        OS << "namespace " << Rest << " {\n";
    } else {
      OS << "namespace " << Name << " {\n\n";
    }
  }

  NamespaceEmitter(const NamespaceEmitter &) = delete;
  NamespaceEmitter &operator=(const NamespaceEmitter &) = delete;

  ~NamespaceEmitter() {
    if (Name.empty())
      return;
    StringRef First, Rest;
    std::tie(First, Rest) = StringRef(Name).split("::");
    switch (MacroKind) {
    case Kind::LLVM:
      if (!Rest.empty())
        OS << "\n} // namespace " << Rest << "\n";
      OS << "LLVM_NAMESPACE_END // namespace llvm\n";
      break;
    case Kind::MLIR:
      if (!Rest.empty())
        OS << "\n} // namespace " << Rest << "\n";
      OS << "MLIR_NAMESPACE_END // namespace mlir\n";
      break;
    case Kind::Plain:
      OS << "\n} // namespace " << Name << "\n";
      break;
    }
  }

private:
  // Trim "::" prefix. If the namespace specified is ""::mlir::toy", then the
  // generated namespace scope needs to use
  //
  // namespace mlir::toy {
  // }
  //
  // and cannot use "namespace ::mlir::toy".
  static StringRef trim(StringRef Name) {
    Name.consume_front("::");
    return Name;
  }

  enum class Kind { Plain, LLVM, MLIR };
  Kind MacroKind = Kind::Plain;
  std::string Name;
  raw_ostream &OS;
};

// Simple RAII helper for emitting anonymous namespace scope.
class AnonNamespaceEmitter {
public:
  AnonNamespaceEmitter(raw_ostream &OS) : OS(OS) { OS << "namespace {\n\n"; }
  AnonNamespaceEmitter(const AnonNamespaceEmitter &) = delete;
  AnonNamespaceEmitter &operator=(const AnonNamespaceEmitter &) = delete;
  ~AnonNamespaceEmitter() { OS << "} // namespace\n"; }

private:
  raw_ostream &OS;
};

LLVM_NAMESPACE_END // end namespace llvm

#endif // LLVM_TABLEGEN_CODEGENHELPERS_H

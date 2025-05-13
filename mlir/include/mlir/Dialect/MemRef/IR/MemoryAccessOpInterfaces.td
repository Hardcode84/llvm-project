//===-- MemoryAccessOpInterfaces.td ---------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MEMREF_MEMORY_ACCESS_OP_INTERFACES
#define MEMREF_MEMORY_ACCESS_OP_INTERFACES

include "mlir/IR/OpBase.td"

def IndexedAccessOpInterface : OpInterface<"IndexedAccessOpInterface"> {
  let description = [{
    An interface for operations that operate on (by loading from or
    storing to, atomically modifying, or otherwise) memory located at an
    index within a memref whose semantics don't depend on the indexing scheme.

    That is, a direct access op is one where, if `%b[%j0, %j1, ..., %jL]`
    points to the same memory as `%a[%i0,%i1, ... %iK]`, it would be
    trivial to replace `%a[%i0, ..., %iK]` with %b[%j0, ... %jL]`.

    Operations may impose constaints on allowable reindexings.
    Returning a non-empty result from `getAccessedShape()` imposes constraints
    on the dimensions whose strides need to be preserved.

    This interface is intended to enable transformations such as folding in
    aliasing operations (like memref.subview or memref.collapse_shape) or
    linearizing memrefs (making them 1-D) to be generic over in-tree and
    out-of-tree operations.
  }];
  let cppNamespace = "::mlir::memref";
  let methods = [
    InterfaceMethod<
      /*desc=*/[{
        Return the memref that this operation accesses.
      }],
      /*retType=*/"::mlir::TypedValue<::mlir::MemRefType>",
      /*methodName=*/"getMemref",
      /*args=*/(ins)>,
    InterfaceMethod<
      /*desc=*/[{
        Return the indices that are used to access the memref returned by getMemref().

        The size of this range must be equal to the rank of the memref returned by
        getMemref().
      }],
      /*retType=*/"::mlir::ValueRange",
      /*methodName=*/"getIndices",
      /*args=*/(ins)>,
    InterfaceMethod<
      /*desc=*/[{
        Return the shape of the portion of the memref that is being accessed by
        this operation, if known, ignoring leading unit dimensions.
        Reindexing transformations may not modify the *strides* of the tlaining
        N dimensions, where N is the size returned value.
      }],
      /*retType=*/"::llvm::SmallVector<int64_t>",
      /*methodName=*/"getAccessedShape",
      /*args=*/(ins),
      /*methodBody=*/[{}],
      /*defaultImplementation=*/[{
        return ::llvm::SmallVector<int64_t>{};
      }]>,
    InterfaceMethod<
      /*desc=*/[{
        Return a clone of this operation with its memref replaced by
        `newMemref` and indices replaced by `newIndices`, leaving other arguments
        and properties unchanged unless they'd be invalidated by the change.

        The caller must ensure that the new memref/index pair points to the same
        location in memory as the values being replaced.

        The element types of the memref may not change.
      }],
      /*retType=*/"::mlir::Operation *",
      /*methodName=*/"cloneWithReindex",
      /*args=*/(ins "::mlir::RewriterBase&":$rewriter, "::mlir::Value":$newMemref,
        "::mlir::ValueRange":$newIndices)
    >,
    InterfaceMethod<
      /*desc=*/[{
        Return true if, either by definition or due to some attribute,
        it's known that all indices are non-negative and less than the size
        of the dimension they index.
      }],
      /*retType=*/"bool",
      /*methodName=*/"hasInboundsIndices",
      /*args=*/(ins),
      /*methodBody=*/[{}],
      /*defaultImplementation=*/[{
        return true;
      }]>
  ];
}

def IndexedMemCopyOpInterface : OpInterface<"IndexedMemCopyOpInterface"> {
  let description = [{
    This is an interface for operations that perform a copy of some number
    of values from `%src[%srcIndices...]` (or some consistently related
    location) to `%dst[%dstIndices...]` (or some consistently related location
    - for example, the destination element may be offset by a lane ID in
    a GPU subgroup).

    The motivating examples for this interface are operations that perform
    direct loads to workgroup memory on GPUs.

    This allows patterns that reindex memrefs (like folding in subview operations)
    to treach such operations as a class, just like `IndexedAccessOpInterface`.

    Unlike `IndexedAccessOpInterface`, this interface assumes that the elements
    being copied are contiguous in memory and that the produces of the
    operation has ensured this. That is, if the source memref is a
    `memref<8x3xf32, strided<[9, 1]>>`, it is presumed that a memcopy of
    4 floats starting at the indices `[%x, %y]` is intended to read into
    the space between the length-three rows.
  }];
  let cppNamespace = "::mlir::memref";
  let methods = [
    InterfaceMethod<
      /*desc=*/[{
        Return the source memref for this copy operation.
      }],
      /*retType=*/"::mlir::TypedValue<::mlir::MemRefType>",
      /*methodName=*/"getSrc",
      /*args=*/(ins)>,
    InterfaceMethod<
      /*desc=*/[{
        Return the indices that are used to access the source memref.

        The size of this range must be equal to the rank of the memref returned by
        getSrc().
      }],
      /*retType=*/"::mlir::ValueRange",
      /*methodName=*/"getSrcIndices",
      /*args=*/(ins)>,
    InterfaceMethod<
      /*desc=*/[{
        Return the destination memref for this copy operation.
      }],
      /*retType=*/"::mlir::TypedValue<::mlir::MemRefType>",
      /*methodName=*/"getDst",
      /*args=*/(ins)>,
    InterfaceMethod<
      /*desc=*/[{
        Return the indices that are used to access the destination memref.

        The size of this range must be equal to the rank of the memref returned by
        getDst().
      }],
      /*retType=*/"::mlir::ValueRange",
      /*methodName=*/"getDstIndices",
      /*args=*/(ins)>,
    InterfaceMethod<
      /*desc=*/[{
        Return a clone of this operation with `src`, `srcIndices`, `dst`, and
        `dstIndices` replaced by `newSrc`, `newSrcIndices`, `newDst`, and `newDstIndices`
        respectively. Other properties of the operation, such as the size of the copy,
        should be preserved.

        The caller must ensure that the new memref/index pairs point to the same
        location in memory.
      }],
      /*retType=*/"::mlir::Operation *",
      /*methodName=*/"cloneWithReindex",
      /*args=*/(ins "::mlir::RewriterBase&":$rewriter,
        "::mlir::Value":$newSrc, "::mlir::ValueRange":$newSrcIndices,
        "::mlir::Value":$newDst, "::mlir::ValueRange":$newDstIndices)
    >
  ];
}
#endif // MEMREF_MEMORY_ACCESS_OP_INTERFACES

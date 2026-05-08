#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from contextlib import contextmanager

from mlir.ir import (
    Context,
    F16Type,
    F32Type,
    IndexType,
    InsertionPoint,
    IntegerType,
    Location,
    Module,
    Type,
    UnitAttr,
)
from mlir.dialects import arith, func, wave, waveamd


def simd_type(element_type=None, width=32):
    element_type = element_type or i32()
    return Type.parse(f"!wave.simd<{element_type}, {width}>")


def mask_type(width=32):
    return Type.parse(f"!wave.mask<{width}>")


def mem_token_type():
    return Type.parse("!wave.mem.token")


def fragment_type(role, element_type, rows=16, columns=16, wave_size=32, registers=4):
    return Type.parse(
        f"!waveamd.fragment<{role}, {element_type}, {rows}, {columns}, "
        f"{wave_size}, {registers}>"
    )


def ptr_type(element_type=None, address_space="#wave.global"):
    element_type = element_type or i32()
    return Type.parse(f"!wave.ptr<{element_type}, {address_space}>")


def buffer_ptr_type(element_type=None):
    return ptr_type(element_type, "#waveamd.buffer")


def i8():
    return IntegerType.get_signless(8)


def i32():
    return IntegerType.get_signless(32)


class ModuleBuilder:
    """Tiny tracing builder around MLIR Python dialect bindings."""

    def __init__(self):
        self.context = Context()
        self.context.allow_unregistered_dialects = True
        self.location = Location.unknown(context=self.context)
        self.module = None

    def __enter__(self):
        self.context.__enter__()
        self.location.__enter__()
        self.module = Module.create()
        self.ip = InsertionPoint(self.module.body)
        self.ip.__enter__()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.ip.__exit__(exc_type, exc, tb)
        self.location.__exit__(exc_type, exc, tb)
        self.context.__exit__(exc_type, exc, tb)

    @contextmanager
    def function(self, name, inputs, results=(), *, kernel=False):
        op = func.FuncOp(name, (inputs, results))
        if kernel:
            op.attributes["wave.kernel"] = UnitAttr.get()
        block = op.add_entry_block()
        with InsertionPoint(block):
            yield FunctionBuilder(block)
            func.ReturnOp([])


class FunctionBuilder:
    def __init__(self, block):
        self.block = block

    @property
    def args(self):
        return self.block.arguments

    def constant_i32(self, value):
        return arith.ConstantOp(value=value, result=i32()).result

    def constant_index(self, value):
        return arith.ConstantOp(value=value, result=IndexType.get()).result

    def lane_id(self, element_type=None, width=32):
        return wave.LaneIdOp(simd_type(element_type, width)).result

    def splat(self, value, element_type=None, width=32):
        return wave.SplatOp(simd_type(element_type or value.type, width), value).result

    def addi(self, lhs, rhs):
        return wave.BinaryOp(lhs.type, "addi", lhs, rhs).result

    def ptr_add(self, base, offset, result_type=None):
        result_type = result_type or base.type
        return wave.PtrAddOp(result_type, base, offset).result

    def store(self, value, ptr, *, after=None):
        return wave.StoreOp(mem_token_type(), value, ptr, dependency=after).token

    def wait(self, *tokens):
        return wave.WaitOp(tokens)

    def token(self):
        return wave.TokenOp(mem_token_type()).result

    def after(self, *tokens):
        return wave.AfterOp(mem_token_type(), tokens).result

    def join(self, *tokens):
        return wave.JoinOp(mem_token_type(), tokens).result

    def fragment_fill(self, value, fragment_type):
        return waveamd.FragmentFillOp(fragment_type, value).result

    def mma(self, kind, a, b, acc):
        return waveamd.MmaOp(acc.type, kind, a, b, acc).result

    def make_buffer(self, base, range_bytes, result_type=None):
        result_type = result_type or buffer_ptr_type()
        return waveamd.MakeBufferOp(result_type, base, range_bytes).result

    def fragment_store(self, fragment, ptr, *, after=None):
        return waveamd.FragmentStoreOp(
            mem_token_type(), fragment, ptr, dependency=after
        ).token


def module():
    return ModuleBuilder()


__all__ = [
    "F16Type",
    "F32Type",
    "IntegerType",
    "IndexType",
    "ModuleBuilder",
    "buffer_ptr_type",
    "fragment_type",
    "i8",
    "i32",
    "mask_type",
    "mem_token_type",
    "module",
    "ptr_type",
    "simd_type",
]

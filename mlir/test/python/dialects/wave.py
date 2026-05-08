# RUN: %PYTHON %s | FileCheck %s

from mlir.dialects import wave_dsl as w


def run(f):
    print("\nTEST:", f.__name__)
    f()


# CHECK-LABEL: TEST: test_generic_wave_kernel
@run
def test_generic_wave_kernel():
    with w.module() as m:
        with m.function(
            "generic_wave_kernel", [w.ptr_type(w.i32()), w.i32()], kernel=True
        ) as f:
            out, x = f.args
            lane = f.lane_id()
            vx = f.splat(x)
            ptrs = f.ptr_add(out, lane, w.simd_type(w.ptr_type(w.i32())))
            token = f.store(vx, ptrs)
            f.wait(token)
        # CHECK: func.func @generic_wave_kernel
        # CHECK: wave.lane_id
        # CHECK: wave.splat
        # CHECK: wave.store
        # CHECK: wave.wait
        print(m.module)


# CHECK-LABEL: TEST: test_waveamd_matrix_kernel
@run
def test_waveamd_matrix_kernel():
    with w.module() as m:
        with m.function("matrix_kernel", [w.ptr_type(w.i32())], kernel=True) as f:
            (out,) = f.args
            zero = f.constant_i32(0)
            seven = f.constant_i32(7)
            base = f.constant_index(0)
            a_t = w.fragment_type(0, w.i8(), registers=4)
            b_t = w.fragment_type(1, w.i8(), registers=4)
            acc_t = w.fragment_type(2, w.i32(), registers=8)
            a = f.fragment_fill(zero, a_t)
            b = f.fragment_fill(zero, b_t)
            acc = f.fragment_fill(seven, acc_t)
            result = f.mma("wmma.i32.16x16x16.iu8", a, b, acc)
            ptr = f.ptr_add(out, base)
            token = f.fragment_store(result, ptr)
            f.wait(token)
        # CHECK: func.func @matrix_kernel
        # CHECK: waveamd.fragment_fill
        # CHECK: waveamd.mma
        # CHECK: waveamd.fragment_store
        # CHECK: wave.wait
        print(m.module)


# CHECK-LABEL: TEST: test_waveamd_buffer_pointer_type
@run
def test_waveamd_buffer_pointer_type():
    with w.module() as m:
        with m.function("buffer_ptr_kernel", [w.ptr_type(w.i32())], kernel=True) as f:
            (out,) = f.args
            range_bytes = f.constant_i32(128)
            buffer = f.make_buffer(out, range_bytes, w.buffer_ptr_type(w.i32()))
            lane = f.lane_id()
            ptrs = f.ptr_add(buffer, lane, w.simd_type(w.buffer_ptr_type(w.i32())))
        # CHECK: func.func @buffer_ptr_kernel
        # CHECK: waveamd.make_buffer
        # CHECK: !wave.ptr<i32, #waveamd.buffer>
        # CHECK: !wave.simd<!wave.ptr<i32, #waveamd.buffer>, 32>
        print(m.module)


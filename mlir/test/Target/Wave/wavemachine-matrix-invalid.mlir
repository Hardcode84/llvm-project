// RUN: mlir-opt -split-input-file -verify-diagnostics %s

func.func @bad_fill_source(%x: i16) {
  // expected-error @below {{source must be an i32 bit pattern}}
  %a = wave.fragment_fill %x : i16 -> !wave.fragment<0, i8, 16, 16, 32, 4>
  return
}

// -----

func.func @bad_fill_shape(%x: i32) {
  // expected-error @below {{only 16x16 fragments are supported for now}}
  %a = wave.fragment_fill %x : i32 -> !wave.fragment<0, i8, 8, 16, 32, 4>
  return
}

// -----

func.func @bad_mma_kind(%x: i32) {
  %a = wave.fragment_fill %x : i32 -> !wave.fragment<0, i8, 16, 16, 32, 4>
  %b = wave.fragment_fill %x : i32 -> !wave.fragment<1, i8, 16, 16, 32, 4>
  %acc = wave.fragment_fill %x : i32 -> !wave.fragment<2, i32, 16, 16, 32, 8>
  // expected-error @below {{unsupported matrix operation kind}}
  %result = wave.mma "wmma.f32.16x16x32.f16" %a, %b, %acc : !wave.fragment<0, i8, 16, 16, 32, 4>, !wave.fragment<1, i8, 16, 16, 32, 4>, !wave.fragment<2, i32, 16, 16, 32, 8> -> !wave.fragment<2, i32, 16, 16, 32, 8>
  return
}

// -----

func.func @bad_mma_b_role(%x: i32) {
  %a = wave.fragment_fill %x : i32 -> !wave.fragment<0, i8, 16, 16, 32, 4>
  %bad_b = wave.fragment_fill %x : i32 -> !wave.fragment<0, i8, 16, 16, 32, 4>
  %acc = wave.fragment_fill %x : i32 -> !wave.fragment<2, i32, 16, 16, 32, 8>
  // expected-error @below {{B operand must be a 16x16 i8 wave32 fragment with 4 registers}}
  %result = wave.mma "wmma.i32.16x16x16.iu8" %a, %bad_b, %acc : !wave.fragment<0, i8, 16, 16, 32, 4>, !wave.fragment<0, i8, 16, 16, 32, 4>, !wave.fragment<2, i32, 16, 16, 32, 8> -> !wave.fragment<2, i32, 16, 16, 32, 8>
  return
}

// -----

func.func @bad_fragment_store_memref(%out: memref<256xindex>, %x: i32) {
  %base = arith.constant 0 : index
  %acc = wave.fragment_fill %x : i32 -> !wave.fragment<2, i32, 16, 16, 32, 8>
  // expected-error @below {{fragment stores currently require a 32-bit memref}}
  wave.fragment_store %acc -> %out[%base] : (!wave.fragment<2, i32, 16, 16, 32, 8>, memref<256xindex>, index) -> ()
  return
}

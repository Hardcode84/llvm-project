//===- wave_buffer_runner.cpp - Wave buffer hardware smoke test -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

static void check(hipError_t error, const char *what) {
  if (error != hipSuccess) {
    std::fprintf(stderr, "%s failed: %s\n", what, hipGetErrorString(error));
    std::exit(1);
  }
}

int main(int argc, char **argv) {
  if (argc != 2) {
    std::fprintf(stderr, "usage: %s <hsaco>\n", argv[0]);
    return 2;
  }

  hipDeviceProp_t props;
  check(hipGetDeviceProperties(&props, 0), "hipGetDeviceProperties");
  std::printf("device: %s (%s)\n", props.name, props.gcnArchName);

  hipModule_t module = nullptr;
  hipFunction_t kernel = nullptr;
  check(hipModuleLoad(&module, argv[1]), "hipModuleLoad");
  check(hipModuleGetFunction(&kernel, module, "buffer_store_kernel"),
        "hipModuleGetFunction");

  constexpr int lanes = 32;
  constexpr int x = 7;
  int32_t *deviceOut = nullptr;
  check(hipMalloc(&deviceOut, lanes * sizeof(int32_t)), "hipMalloc");
  check(hipMemset(deviceOut, 0, lanes * sizeof(int32_t)), "hipMemset");

  void *args[] = {&deviceOut, const_cast<int *>(&x)};
  check(hipModuleLaunchKernel(kernel, 1, 1, 1, lanes, 1, 1, 0, nullptr, args,
                              nullptr),
        "hipModuleLaunchKernel");
  check(hipDeviceSynchronize(), "hipDeviceSynchronize");

  std::vector<int32_t> hostOut(lanes);
  check(hipMemcpy(hostOut.data(), deviceOut, lanes * sizeof(int32_t),
                  hipMemcpyDeviceToHost),
        "hipMemcpy");

  bool ok = true;
  for (int i = 0; i != lanes; ++i) {
    int32_t expected = i + x;
    if (hostOut[i] != expected) {
      std::fprintf(stderr, "mismatch lane %d: got %d expected %d\n", i,
                   hostOut[i], expected);
      ok = false;
    }
  }

  check(hipFree(deviceOut), "hipFree");
  check(hipModuleUnload(module), "hipModuleUnload");

  if (!ok)
    return 1;
  std::printf("buffer_store_kernel ok: lanes 0..31 plus %d verified\n", x);
  return 0;
}

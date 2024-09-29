//===-- Loader Implementation for NVPTX devices --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file impelements a simple loader to run images supporting the NVPTX
// architecture. The file launches the '_start' kernel which should be provided
// by the device application start code and call ultimately call the 'main'
// function.
//
//===----------------------------------------------------------------------===//

#include "Loader.h"

#include "cuda.h"

#include "llvm/Object/ELF.h"
#include "llvm/Object/ELFObjectFile.h"

#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <SDL.h>

using namespace llvm;
using namespace object;

static void handle_error_impl(const char *file, int32_t line, CUresult err) {
  if (err == CUDA_SUCCESS)
    return;

  const char *err_str = nullptr;
  CUresult result = cuGetErrorString(err, &err_str);
  if (result != CUDA_SUCCESS)
    fprintf(stderr, "%s:%d:0: Unknown Error\n", file, line);
  else
    fprintf(stderr, "%s:%d:0: Error: %s\n", file, line, err_str);
  exit(1);
}

// Gets the names of all the globals that contain functions to initialize or
// deinitialize. We need to do this manually because the NVPTX toolchain does
// not contain the necessary binary manipulation tools.
template <typename Alloc>
Expected<void *> get_ctor_dtor_array(const void *image, const size_t size,
                                     Alloc allocator, CUmodule binary) {
  auto mem_buffer = MemoryBuffer::getMemBuffer(
      StringRef(reinterpret_cast<const char *>(image), size), "image",
      /*RequiresNullTerminator=*/false);
  Expected<ELF64LEObjectFile> elf_or_err =
      ELF64LEObjectFile::create(*mem_buffer);
  if (!elf_or_err)
    handle_error(toString(elf_or_err.takeError()).c_str());

  std::vector<std::pair<const char *, uint16_t>> ctors;
  std::vector<std::pair<const char *, uint16_t>> dtors;
  // CUDA has no way to iterate over all the symbols so we need to inspect the
  // ELF directly using the LLVM libraries.
  for (const auto &symbol : elf_or_err->symbols()) {
    auto name_or_err = symbol.getName();
    if (!name_or_err)
      handle_error(toString(name_or_err.takeError()).c_str());

    // Search for all symbols that contain a constructor or destructor.
    if (!name_or_err->starts_with("__init_array_object_") &&
        !name_or_err->starts_with("__fini_array_object_"))
      continue;

    uint16_t priority;
    if (name_or_err->rsplit('_').second.getAsInteger(10, priority))
      handle_error("Invalid priority for constructor or destructor");

    if (name_or_err->starts_with("__init"))
      ctors.emplace_back(std::make_pair(name_or_err->data(), priority));
    else
      dtors.emplace_back(std::make_pair(name_or_err->data(), priority));
  }
  // Lower priority constructors are run before higher ones. The reverse is true
  // for destructors.
  llvm::sort(ctors, [](auto x, auto y) { return x.second < y.second; });
  llvm::sort(dtors, [](auto x, auto y) { return x.second < y.second; });

  // Allocate host pinned memory to make these arrays visible to the GPU.
  CUdeviceptr *dev_memory = reinterpret_cast<CUdeviceptr *>(allocator(
      ctors.size() * sizeof(CUdeviceptr) + dtors.size() * sizeof(CUdeviceptr)));
  uint64_t global_size = 0;

  // Get the address of the global and then store the address of the constructor
  // function to call in the constructor array.
  CUdeviceptr *dev_ctors_start = dev_memory;
  CUdeviceptr *dev_ctors_end = dev_ctors_start + ctors.size();
  for (uint64_t i = 0; i < ctors.size(); ++i) {
    CUdeviceptr dev_ptr;
    if (CUresult err =
            cuModuleGetGlobal(&dev_ptr, &global_size, binary, ctors[i].first))
      handle_error(err);
    if (CUresult err =
            cuMemcpyDtoH(&dev_ctors_start[i], dev_ptr, sizeof(uintptr_t)))
      handle_error(err);
  }

  // Get the address of the global and then store the address of the destructor
  // function to call in the destructor array.
  CUdeviceptr *dev_dtors_start = dev_ctors_end;
  CUdeviceptr *dev_dtors_end = dev_dtors_start + dtors.size();
  for (uint64_t i = 0; i < dtors.size(); ++i) {
    CUdeviceptr dev_ptr;
    if (CUresult err =
            cuModuleGetGlobal(&dev_ptr, &global_size, binary, dtors[i].first))
      handle_error(err);
    if (CUresult err =
            cuMemcpyDtoH(&dev_dtors_start[i], dev_ptr, sizeof(uintptr_t)))
      handle_error(err);
  }

  // Obtain the address of the pointers the startup implementation uses to
  // iterate the constructors and destructors.
  CUdeviceptr init_start;
  if (CUresult err = cuModuleGetGlobal(&init_start, &global_size, binary,
                                       "__init_array_start"))
    handle_error(err);
  CUdeviceptr init_end;
  if (CUresult err = cuModuleGetGlobal(&init_end, &global_size, binary,
                                       "__init_array_end"))
    handle_error(err);
  CUdeviceptr fini_start;
  if (CUresult err = cuModuleGetGlobal(&fini_start, &global_size, binary,
                                       "__fini_array_start"))
    handle_error(err);
  CUdeviceptr fini_end;
  if (CUresult err = cuModuleGetGlobal(&fini_end, &global_size, binary,
                                       "__fini_array_end"))
    handle_error(err);

  // Copy the pointers to the newly written array to the symbols so the startup
  // implementation can iterate them.
  if (CUresult err =
          cuMemcpyHtoD(init_start, &dev_ctors_start, sizeof(uintptr_t)))
    handle_error(err);
  if (CUresult err = cuMemcpyHtoD(init_end, &dev_ctors_end, sizeof(uintptr_t)))
    handle_error(err);
  if (CUresult err =
          cuMemcpyHtoD(fini_start, &dev_dtors_start, sizeof(uintptr_t)))
    handle_error(err);
  if (CUresult err = cuMemcpyHtoD(fini_end, &dev_dtors_end, sizeof(uintptr_t)))
    handle_error(err);

  return dev_memory;
}

void print_kernel_resources(CUmodule binary, const char *kernel_name) {
  CUfunction function;
  if (CUresult err = cuModuleGetFunction(&function, binary, kernel_name))
    handle_error(err);
  int num_regs;
  if (CUresult err =
          cuFuncGetAttribute(&num_regs, CU_FUNC_ATTRIBUTE_NUM_REGS, function))
    handle_error(err);
  printf("Executing kernel %s:\n", kernel_name);
  printf("%6s registers: %d\n", kernel_name, num_regs);
}

namespace {
struct BumpPtrAlloc {
  uintptr_t prealloc_current = 0;
  uintptr_t prealloc_end = 0;

  uintptr_t alloc(size_t size, size_t align) {
    if ((prealloc_current + (size + align)) <= prealloc_end) {
      uintptr_t ptr = (prealloc_current + align - 1) & ~(align - 1);
      prealloc_current += (size + align);
      return ptr;
    }
    return 0;
  }

  bool is_allocated(uintptr_t ptr) const {
    return ptr >= prealloc_current && ptr < prealloc_end;
  }
};
}

template <typename T>
static T read_global_var(CUmodule binary, CUstream stream, const char *name) {
  size_t size = 0;
  CUdeviceptr ptr;
  if (CUresult err = cuModuleGetGlobal(&ptr, &size, binary, name))
    handle_error(err);

  if (size != sizeof(T)) {
    fprintf(stderr, "Invalid var %s size, expected %d got %d\n", name,
            int(sizeof(T)), int(size));
    exit(EXIT_FAILURE);
  }

  T ret;

  if (CUresult err = cuMemcpyDtoHAsync(&ret, ptr, size, stream))
    handle_error(err);

  if (CUresult err = cuStreamSynchronize(stream))
    handle_error(err);

  return ret;
}

template <typename T>
static void write_global_var(CUmodule binary, CUstream stream, const char *name,
                             const T &val) {
  size_t size = 0;
  CUdeviceptr ptr;
  if (CUresult err = cuModuleGetGlobal(&ptr, &size, binary, name))
    handle_error(err);

  if (size != sizeof(T)) {
    fprintf(stderr, "Invalid var %s size, expected %d got %d\n", name,
            int(sizeof(T)), int(size));
    exit(EXIT_FAILURE);
  }

  if (CUresult err = cuMemcpyHtoDAsync(ptr, &val, size, stream))
    handle_error(err);

  if (CUresult err = cuStreamSynchronize(stream))
    handle_error(err);
}

static bool handle_rpc_server(rpc_device_t rpc_device, CUstream stream) {
  if (cuStreamQuery(stream) != CUDA_ERROR_NOT_READY)
    return false;

  if (rpc_status_t err = rpc_handle_server(rpc_device))
      handle_error(err);

    return true;
}

static int HostInsideBarrier = 0;
namespace {
struct HostBarrierGuard {
  HostBarrierGuard() { ++HostInsideBarrier; }
  ~HostBarrierGuard() { --HostInsideBarrier; }
};
} // namespace

static int GpuBlocksInsideBarrier = 0;
static size_t gpu_barrier_enter(void *) {
  if (HostInsideBarrier == 0)
    return 0;

  ++GpuBlocksInsideBarrier;
  return 1;
}

static size_t gpu_barrier_exit(void *) {
  if (HostInsideBarrier != 0)
    return 0;

  --GpuBlocksInsideBarrier;
  return 1;
}

static bool handle_rpc_barrier_impl(rpc_device_t rpc_device, CUstream stream,
                                    uint32_t num_blocks) {
  {
    HostBarrierGuard g;
    while (true) {
      if (cuStreamQuery(stream) != CUDA_ERROR_NOT_READY)
        return false;

      if (rpc_status_t err = rpc_handle_server(rpc_device))
        handle_error(err);

      if ((uint32_t)GpuBlocksInsideBarrier == num_blocks)
        break;
    }
  }
  do {
    if (!handle_rpc_server(rpc_device, stream))
      return false;
  } while (GpuBlocksInsideBarrier != 0);

  return true;
}

template <typename args_t>
CUresult launch_kernel(CUmodule binary, CUstream stream, BumpPtrAlloc& alloc,
                       rpc_device_t rpc_device, const LaunchParameters &params,
                       const char *kernel_name, args_t kernel_args,
                       bool print_resource_usage) {
  // look up the '_start' kernel in the loaded module.
  CUfunction function;
  if (CUresult err = cuModuleGetFunction(&function, binary, kernel_name))
    handle_error(err);

  // Set up the arguments to the '_start' kernel on the GPU.
  uint64_t args_size = sizeof(args_t);
  void *args_config[] = {CU_LAUNCH_PARAM_BUFFER_POINTER, &kernel_args,
                         CU_LAUNCH_PARAM_BUFFER_SIZE, &args_size,
                         CU_LAUNCH_PARAM_END};

  // Initialize a non-blocking CUDA stream to allocate memory if needed. This
  // needs to be done on a separate stream or else it will deadlock with the
  // executing kernel.
  CUstream memory_stream;
  if (CUresult err = cuStreamCreate(&memory_stream, CU_STREAM_NON_BLOCKING))
    handle_error(err);

  // Register RPC callbacks for the malloc and free functions on HSA.
  register_rpc_callbacks<32>(rpc_device);

  rpc_register_callback(
      rpc_device, RPC_MALLOC,
      [](rpc_port_t port, void *data) {
        auto malloc_handler = [](rpc_buffer_t *buffer, void *data) -> void {
          uint64_t size = buffer->data[0];
          auto* alloc = static_cast<BumpPtrAlloc*>(data);
          if (auto ptr = alloc->alloc(size, 16)) {
            // printf("alloc arena %d %p\n", (int)size, (void*)ptr);
            buffer->data[0] = static_cast<uintptr_t>(ptr);
            return;
          }

          CUdeviceptr dev_ptr;
          if (CUresult err = cuMemAlloc(&dev_ptr, size))
            dev_ptr = 0UL;

          // printf("alloc %d %p\n", (int)size, (void*)dev_ptr);
          buffer->data[0] = static_cast<uintptr_t>(dev_ptr);
        };
        rpc_recv_and_send(port, malloc_handler, data);
      },
      &alloc);
  rpc_register_callback(
      rpc_device, RPC_FREE,
      [](rpc_port_t port, void *data) {
        auto free_handler = [](rpc_buffer_t *buffer, void *data) {
          uint64_t ptr = buffer->data[0];
          auto* alloc = static_cast<BumpPtrAlloc*>(data);
          if (alloc->is_allocated(ptr)) {
            // printf("free arena %p\n", (void*)ptr);
            return;
          }

          if (CUresult err = cuMemFree(
                  static_cast<CUdeviceptr>(ptr)))
            handle_error(err);

          // printf("free %p\n", (void*)ptr);
        };
        rpc_recv_and_send(port, free_handler, data);
      },
      &alloc);

  if (print_resource_usage)
    print_kernel_resources(binary, kernel_name);

  // Call the kernel with the given arguments.
  if (CUresult err = cuLaunchKernel(
          function, params.num_blocks_x, params.num_blocks_y,
          params.num_blocks_z, params.num_threads_x, params.num_threads_y,
          params.num_threads_z, 0, stream, nullptr, args_config))
    handle_error(err);

  // Wait until the kernel has completed execution on the device. Periodically
  // check the RPC client for work to be performed on the server.
  while (handle_rpc_server(rpc_device, stream));

  // Handle the server one more time in case the kernel exited with a pending
  // send still in flight.
  if (rpc_status_t err = rpc_handle_server(rpc_device))
    handle_error(err);

  return CUDA_SUCCESS;
}

#define KEY_RIGHTARROW 0xae
#define KEY_LEFTARROW 0xac
#define KEY_UPARROW 0xad
#define KEY_DOWNARROW 0xaf
#define KEY_STRAFE_L 0xa0
#define KEY_STRAFE_R 0xa1
#define KEY_USE 0xa2
#define KEY_FIRE 0xa3
#define KEY_ESCAPE 27
#define KEY_ENTER 13
#define KEY_TAB 9
#define KEY_F1 (0x80 + 0x3b)
#define KEY_F2 (0x80 + 0x3c)
#define KEY_F3 (0x80 + 0x3d)
#define KEY_F4 (0x80 + 0x3e)
#define KEY_F5 (0x80 + 0x3f)
#define KEY_F6 (0x80 + 0x40)
#define KEY_F7 (0x80 + 0x41)
#define KEY_F8 (0x80 + 0x42)
#define KEY_F9 (0x80 + 0x43)
#define KEY_F10 (0x80 + 0x44)
#define KEY_F11 (0x80 + 0x57)
#define KEY_F12 (0x80 + 0x58)

#define KEY_BACKSPACE 0x7f
#define KEY_PAUSE 0xff

#define KEY_EQUALS 0x3d
#define KEY_MINUS 0x2d

#define KEY_RSHIFT (0x80 + 0x36)
#define KEY_RCTRL (0x80 + 0x1d)
#define KEY_RALT (0x80 + 0x38)

#define KEY_LALT KEY_RALT

// new keys:

#define KEY_CAPSLOCK (0x80 + 0x3a)
#define KEY_NUMLOCK (0x80 + 0x45)
#define KEY_SCRLCK (0x80 + 0x46)
#define KEY_PRTSCR (0x80 + 0x59)

#define KEY_HOME (0x80 + 0x47)
#define KEY_END (0x80 + 0x4f)
#define KEY_PGUP (0x80 + 0x49)
#define KEY_PGDN (0x80 + 0x51)
#define KEY_INS (0x80 + 0x52)
#define KEY_DEL (0x80 + 0x53)

#define KEYP_0 0
#define KEYP_1 KEY_END
#define KEYP_2 KEY_DOWNARROW
#define KEYP_3 KEY_PGDN
#define KEYP_4 KEY_LEFTARROW
#define KEYP_5 '5'
#define KEYP_6 KEY_RIGHTARROW
#define KEYP_7 KEY_HOME
#define KEYP_8 KEY_UPARROW
#define KEYP_9 KEY_PGUP

#define KEYP_DIVIDE '/'
#define KEYP_PLUS '+'
#define KEYP_MINUS '-'
#define KEYP_MULTIPLY '*'
#define KEYP_PERIOD 0
#define KEYP_EQUALS KEY_EQUALS
#define KEYP_ENTER KEY_ENTER

static unsigned char convertToDoomKey(unsigned int key) {
  switch (key) {
  case SDLK_RETURN:
    key = KEY_ENTER;
    break;
  case SDLK_ESCAPE:
    key = KEY_ESCAPE;
    break;
  case SDLK_LEFT:
    key = KEY_LEFTARROW;
    break;
  case SDLK_RIGHT:
    key = KEY_RIGHTARROW;
    break;
  case SDLK_UP:
    key = KEY_UPARROW;
    break;
  case SDLK_DOWN:
    key = KEY_DOWNARROW;
    break;
  case SDLK_LCTRL:
  case SDLK_RCTRL:
    key = KEY_FIRE;
    break;
  case SDLK_SPACE:
    key = KEY_USE;
    break;
  case SDLK_LSHIFT:
  case SDLK_RSHIFT:
    key = KEY_RSHIFT;
    break;
  case SDLK_LALT:
  case SDLK_RALT:
    key = KEY_LALT;
    break;
  case SDLK_F2:
    key = KEY_F2;
    break;
  case SDLK_F3:
    key = KEY_F3;
    break;
  case SDLK_F4:
    key = KEY_F4;
    break;
  case SDLK_F5:
    key = KEY_F5;
    break;
  case SDLK_F6:
    key = KEY_F6;
    break;
  case SDLK_F7:
    key = KEY_F7;
    break;
  case SDLK_F8:
    key = KEY_F8;
    break;
  case SDLK_F9:
    key = KEY_F9;
    break;
  case SDLK_F10:
    key = KEY_F10;
    break;
  case SDLK_F11:
    key = KEY_F11;
    break;
  case SDLK_EQUALS:
  case SDLK_PLUS:
    key = KEY_EQUALS;
    break;
  case SDLK_MINUS:
    key = KEY_MINUS;
    break;
  default:
    key = tolower(key);
    break;
  }

  return key;
}

static const int KEYQUEUE_SIZE = 16;
#define KeyQueueReadIndex (KeyQueue[KEYQUEUE_SIZE])
#define KeyQueueWriteIndex (KeyQueue[KEYQUEUE_SIZE + 1])
static void addKeyToQueue(unsigned short *KeyQueue, int pressed,
                          unsigned int keyCode) {
  unsigned char key = convertToDoomKey(keyCode);

  unsigned short keyData = (pressed << 8) | key;

  KeyQueue[KeyQueueWriteIndex] = keyData;
  KeyQueueWriteIndex++;
  KeyQueueWriteIndex %= KEYQUEUE_SIZE;
}

static void handleKeyInput(unsigned short *KeyQueue) {
  SDL_Event e;
  while (SDL_PollEvent(&e)) {
    if (e.type == SDL_QUIT) {
      puts("Quit requested");
      exit(1);
    }
    if (e.type == SDL_KEYDOWN) {
      addKeyToQueue(KeyQueue, 1, e.key.keysym.sym);
    } else if (e.type == SDL_KEYUP) {
      addKeyToQueue(KeyQueue, 0, e.key.keysym.sym);
    }
  }
}
#undef KeyQueueReadIndex
#undef KeyQueueWriteIndex

template <typename args_t>
CUresult launch_main_loop(CUmodule binary, CUstream stream, BumpPtrAlloc &alloc,
                          rpc_device_t rpc_device,
                          const LaunchParameters &params,
                          const char *kernel_name, args_t kernel_args,
                          bool print_resource_usage) {
  const size_t keybuffer_size = KEYQUEUE_SIZE + 2;
  unsigned short *key_queue = nullptr;
  if (CUresult err = cuMemAllocHost((void **)&key_queue,
                                    keybuffer_size * (sizeof(key_queue[0]))))
    handle_error(err);

  // look up the '_start' kernel in the loaded module.
  CUfunction function;
  if (CUresult err = cuModuleGetFunction(&function, binary, kernel_name))
    handle_error(err);

  // Set up the arguments to the '_start' kernel on the GPU.
  uint64_t args_size = sizeof(args_t);
  void *args_config[] = {CU_LAUNCH_PARAM_BUFFER_POINTER, &kernel_args,
                         CU_LAUNCH_PARAM_BUFFER_SIZE, &args_size,
                         CU_LAUNCH_PARAM_END};

  // Initialize a non-blocking CUDA stream to allocate memory if needed. This
  // needs to be done on a separate stream or else it will deadlock with the
  // executing kernel.
  CUstream memory_stream;
  if (CUresult err = cuStreamCreate(&memory_stream, CU_STREAM_NON_BLOCKING))
    handle_error(err);

  write_global_var(binary, memory_stream, "DG_GPU_KeyQueue", key_queue);

  write_global_var(binary, memory_stream, "DG_GPU_BarrierEnter",
                   (void *)&gpu_barrier_enter);
  write_global_var(binary, memory_stream, "DG_GPU_BarrierExit",
                   (void *)&gpu_barrier_exit);

  // Register RPC callbacks for the malloc and free functions on HSA.
  register_rpc_callbacks<32>(rpc_device);

  rpc_register_callback(
      rpc_device, RPC_MALLOC,
      [](rpc_port_t port, void *data) {
        auto malloc_handler = [](rpc_buffer_t *buffer, void *data) -> void {
          uint64_t size = buffer->data[0];
          auto *alloc = static_cast<BumpPtrAlloc *>(data);
          if (auto ptr = alloc->alloc(size, 16)) {
            // printf("alloc arena %d %p\n", (int)size, (void*)ptr);
            buffer->data[0] = static_cast<uintptr_t>(ptr);
            return;
          }

          CUdeviceptr dev_ptr;
          if (CUresult err = cuMemAlloc(&dev_ptr, size))
            dev_ptr = 0UL;

          // printf("alloc %d %p\n", (int)size, (void*)dev_ptr);
          buffer->data[0] = static_cast<uintptr_t>(dev_ptr);
        };
        rpc_recv_and_send(port, malloc_handler, data);
      },
      &alloc);
  rpc_register_callback(
      rpc_device, RPC_FREE,
      [](rpc_port_t port, void *data) {
        auto free_handler = [](rpc_buffer_t *buffer, void *data) {
          uint64_t ptr = buffer->data[0];
          auto *alloc = static_cast<BumpPtrAlloc *>(data);
          if (alloc->is_allocated(ptr)) {
            // printf("free arena %p\n", (void*)ptr);
            return;
          }

          if (CUresult err = cuMemFree(static_cast<CUdeviceptr>(ptr)))
            handle_error(err);

          // printf("free %p\n", (void*)ptr);
        };
        rpc_recv_and_send(port, free_handler, data);
      },
      &alloc);

  if (print_resource_usage)
    print_kernel_resources(binary, kernel_name);

  // Call the kernel with the given arguments.
  if (CUresult err = cuLaunchKernel(
          function, params.num_blocks_x, params.num_blocks_y,
          params.num_blocks_z, params.num_threads_x, params.num_threads_y,
          params.num_threads_z, 0, stream, nullptr, args_config))
    handle_error(err);

  auto num_blocks =
      params.num_blocks_x * params.num_blocks_y * params.num_blocks_z;
  auto handle_rpc_barrier = [&]() -> bool {
    return handle_rpc_barrier_impl(rpc_device, stream, num_blocks);
  };

  // After init barrier
  handle_rpc_barrier();

  auto screen_w = read_global_var<int>(binary, memory_stream, "DG_GPU_ResX");
  auto screen_h = read_global_var<int>(binary, memory_stream, "DG_GPU_Resy");
  auto screen_ptr = read_global_var<CUdeviceptr>(binary, memory_stream,
                                                 "DG_GPU_ScreenBuffer");

  printf("Screen: %dx%d %p\n", screen_w, screen_h, (void *)screen_ptr);

  auto window =
      SDL_CreateWindow("DOOM", SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED,
                       screen_w, screen_h, SDL_WINDOW_SHOWN);

  // Setup renderer
  auto renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED);
  // Clear winow
  SDL_RenderClear(renderer);
  // Render the rect to the screen
  SDL_RenderPresent(renderer);

  auto texture =
      SDL_CreateTexture(renderer, SDL_PIXELFORMAT_RGB888,
                        SDL_TEXTUREACCESS_TARGET, screen_w, screen_h);

  size_t screenbuffer_size = screen_w * screen_h * sizeof(uint32_t);
  std::unique_ptr<char[]> temp_screen(new char[screenbuffer_size]);

  while (true) {
    if (!handle_rpc_barrier())
      break;

    handleKeyInput(key_queue);

    // Post draw barrier
    if (!handle_rpc_barrier())
      break;

    if (CUresult err = cuMemcpyDtoHAsync(temp_screen.get(), screen_ptr,
                                         screenbuffer_size, memory_stream))
      handle_error(err);

    if (CUresult err = cuStreamSynchronize(memory_stream))
      handle_error(err);

    SDL_UpdateTexture(texture, nullptr, temp_screen.get(),
                      screen_w * sizeof(uint32_t));

    SDL_RenderClear(renderer);
    SDL_RenderCopy(renderer, texture, nullptr, nullptr);
    SDL_RenderPresent(renderer);
  }

  // Handle the server one more time in case the kernel exited with a pending
  // send still in flight.
  if (rpc_status_t err = rpc_handle_server(rpc_device))
    handle_error(err);

  return CUDA_SUCCESS;
}

int load(int argc, const char **argv, const char **envp, void *image,
         size_t size, const LaunchParameters &params,
         bool print_resource_usage) {
  if (CUresult err = cuInit(0))
    handle_error(err);
  // Obtain the first device found on the system.
  uint32_t device_id = 0;
  CUdevice device;
  if (CUresult err = cuDeviceGet(&device, device_id))
    handle_error(err);

  // Initialize the CUDA context and claim it for this execution.
  CUcontext context;
  if (CUresult err = cuDevicePrimaryCtxRetain(&context, device))
    handle_error(err);
  if (CUresult err = cuCtxSetCurrent(context))
    handle_error(err);

  // Increase the stack size per thread.
  // TODO: We should allow this to be passed in so only the tests that require a
  // larger stack can specify it to save on memory usage.
  if (CUresult err = cuCtxSetLimit(CU_LIMIT_STACK_SIZE, 3 * 1024))
    handle_error(err);

  // Initialize a non-blocking CUDA stream to execute the kernel.
  CUstream stream;
  if (CUresult err = cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING))
    handle_error(err);

  // Load the image into a CUDA module.
  CUmodule binary;
  if (CUresult err = cuModuleLoadDataEx(&binary, image, 0, nullptr, nullptr))
    handle_error(err);

  // Allocate pinned memory on the host to hold the pointer array for the
  // copied argv and allow the GPU device to access it.
  auto allocator = [&](uint64_t size) -> void * {
    void *dev_ptr;
    if (CUresult err = cuMemAllocHost(&dev_ptr, size))
      handle_error(err);
    return dev_ptr;
  };

  auto memory_or_err = get_ctor_dtor_array(image, size, allocator, binary);
  if (!memory_or_err)
    handle_error(toString(memory_or_err.takeError()).c_str());

  void *dev_argv = copy_argument_vector(argc, argv, allocator);
  if (!dev_argv)
    handle_error("Failed to allocate device argv");

  // Allocate pinned memory on the host to hold the pointer array for the
  // copied environment array and allow the GPU device to access it.
  void *dev_envp = copy_environment(envp, allocator);
  if (!dev_envp)
    handle_error("Failed to allocate device environment");

  // Allocate space for the return pointer and initialize it to zero.
  CUdeviceptr dev_ret;
  if (CUresult err = cuMemAlloc(&dev_ret, sizeof(int)))
    handle_error(err);
  if (CUresult err = cuMemsetD32(dev_ret, 0, 1))
    handle_error(err);

  uint32_t warp_size = 32;
  auto rpc_alloc = [](uint64_t size, void *) -> void * {
    void *dev_ptr;
    if (CUresult err = cuMemAllocHost(&dev_ptr, size))
      handle_error(err);
    return dev_ptr;
  };
  rpc_device_t rpc_device;
  if (rpc_status_t err = rpc_server_init(&rpc_device, RPC_MAXIMUM_PORT_COUNT,
                                         warp_size, rpc_alloc, nullptr))
    handle_error(err);

  // Initialize the RPC client on the device by copying the local data to the
  // device's internal pointer.
  CUdeviceptr rpc_client_dev = 0;
  uint64_t client_ptr_size = sizeof(void *);
  if (CUresult err = cuModuleGetGlobal(&rpc_client_dev, &client_ptr_size,
                                       binary, rpc_client_symbol_name))
    handle_error(err);

  CUdeviceptr rpc_client_host = 0;
  if (CUresult err =
          cuMemcpyDtoH(&rpc_client_host, rpc_client_dev, sizeof(void *)))
    handle_error(err);
  if (CUresult err =
          cuMemcpyHtoD(rpc_client_host, rpc_get_client_buffer(rpc_device),
                       rpc_get_client_size()))
    handle_error(err);

  size_t prealloc_size = 64 * 1024 * 1024;
  CUdeviceptr prealloc_ptr;
  if (CUresult err = cuMemAlloc(&prealloc_ptr, prealloc_size))
    handle_error(err);

  BumpPtrAlloc alloc;
  alloc.prealloc_current = static_cast<uintptr_t>(prealloc_ptr);
  alloc.prealloc_end = alloc.prealloc_current + prealloc_size;

  LaunchParameters single_threaded_params = {1, 1, 1, 1, 1, 1};
  begin_args_t init_args = {argc, dev_argv, dev_envp};
  if (CUresult err =
          launch_kernel(binary, stream, alloc, rpc_device, single_threaded_params,
                        "_begin", init_args, print_resource_usage))
    handle_error(err);

  start_args_t args = {argc, dev_argv, dev_envp,
                       reinterpret_cast<void *>(dev_ret)};
  if (CUresult err = launch_main_loop(binary, stream, alloc, rpc_device, params,
                                      "_start", args, print_resource_usage))
    handle_error(err);

  // Copy the return value back from the kernel and wait.
  int host_ret = 0;
  if (CUresult err = cuMemcpyDtoH(&host_ret, dev_ret, sizeof(int)))
    handle_error(err);

  if (CUresult err = cuStreamSynchronize(stream))
    handle_error(err);

  end_args_t fini_args = {host_ret};
  if (CUresult err =
          launch_kernel(binary, stream, alloc, rpc_device, single_threaded_params,
                        "_end", fini_args, print_resource_usage))
    handle_error(err);

  // Free the memory allocated for the device.
  if (CUresult err = cuMemFreeHost(*memory_or_err))
    handle_error(err);
  if (CUresult err = cuMemFree(dev_ret))
    handle_error(err);
  if (CUresult err = cuMemFreeHost(dev_argv))
    handle_error(err);
  if (rpc_status_t err = rpc_server_shutdown(
          rpc_device, [](void *ptr, void *) { cuMemFreeHost(ptr); }, nullptr))
    handle_error(err);

  // Destroy the context and the loaded binary.
  if (CUresult err = cuModuleUnload(binary))
    handle_error(err);
  if (CUresult err = cuDevicePrimaryCtxRelease(device))
    handle_error(err);
  return host_ret;
}

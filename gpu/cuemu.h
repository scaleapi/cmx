#pragma once
// A small CPU emulator for the kernels of kernels.cu, to test the GPU code paths on a machine without a GPU.
// Each thread of a thread block is an OS thread; blocks run one after another.  The warp intrinsics are
// collective operations on a per-warp barrier, so they need every lane of the warp to call them, which the
// kernels do (their warp-level loops are warp-uniform).  Shared memory is static (one block at a time);
// dynamic shared memory is a per-launch buffer.  Atomics use std::atomic_ref.
//
// (From the bf16ref repository's prototype/gpu/cuemu.cpp.)  bf16ref_gpu.cu includes this header instead of the CUDA
// runtime when built with -DBFR_GPU_EMU (C++20), so that its host code and the kernels run on a machine
// without a GPU: emu_run(kernel, grid, block, smem, args...) runs the kernel's blocks one after another.
#include <atomic>
#include <barrier>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <map>
#include <string>
#include <thread>
#include <utility>
#include <vector>

struct dim3e { unsigned x = 1, y = 1, z = 1; };
static thread_local dim3e threadIdx, blockIdx, blockDim;
static thread_local unsigned char *emu_dsmem;

struct Warp {
    std::barrier<> bar{32};
    unsigned long long slot[32];
};
static thread_local Warp *emu_warp;
static thread_local std::barrier<> *emu_block_bar;

#define __global__
#define __device__
#define __host__
#define __shared__ static
#define __forceinline__ inline
#define __restrict__
#define G8_EMU 1
#define DSMEM(T, name) T *name = (T *)emu_dsmem
struct uint4 { unsigned x, y, z, w; };

static inline unsigned lane_id() { return threadIdx.x & 31; }
static inline unsigned lanemask_lt() { return (1u << lane_id()) - 1; }
static inline unsigned lanemask_gt() { return lane_id() == 31 ? 0u : ~((2u << lane_id()) - 1); }
template <class T> static inline T __ldg(const T *p) { return *p; }
static inline unsigned __funnelshift_r(unsigned lo, unsigned hi, unsigned s) { return (unsigned)((((unsigned long long)hi << 32) | lo) >> (s & 31)); }
static inline int __popc(unsigned x) { return __builtin_popcount(x); }
static inline int __ffs(int x) { return __builtin_ffs(x); }
static inline int __clz(int x) { return x ? __builtin_clz((unsigned)x) : 32; }
static inline void __syncthreads() { emu_block_bar->arrive_and_wait(); }
static inline void __syncwarp(unsigned = 0xFFFFFFFFu) { emu_warp->bar.arrive_and_wait(); }
template <class T> static T exch(T v, int src) {            // every lane publishes v and reads lane src's
    Warp &w = *emu_warp; unsigned l = lane_id();
    unsigned long long u = 0; memcpy(&u, &v, sizeof v); w.slot[l] = u;
    w.bar.arrive_and_wait();
    unsigned long long r = w.slot[src]; w.bar.arrive_and_wait();
    T out; memcpy(&out, &r, sizeof out); return out;
}
static std::vector<unsigned long long> gather_all(unsigned long long v) {
    Warp &w = *emu_warp; w.slot[lane_id()] = v; w.bar.arrive_and_wait();
    std::vector<unsigned long long> a(w.slot, w.slot + 32); w.bar.arrive_and_wait(); return a;
}
static inline unsigned __ballot_sync(unsigned mask, int p) {
    auto a = gather_all(p != 0); unsigned r = 0; for (int i = 0; i < 32; i++) if (a[i]) r |= 1u << i; return r & mask;
}
static inline unsigned __activemask() { return 0xFFFFFFFFu; }
static inline unsigned __match_any_sync(unsigned, unsigned v) {
    auto a = gather_all(v); unsigned r = 0; for (int i = 0; i < 32; i++) if (a[i] == v) r |= 1u << i; return r;
}
static inline unsigned __reduce_max_sync(unsigned, unsigned v) { auto a = gather_all(v); unsigned m = 0; for (auto x : a) m = x > m ? (unsigned)x : m; return m; }
static inline unsigned __reduce_add_sync(unsigned, unsigned v) { auto a = gather_all(v); unsigned m = 0; for (auto x : a) m += (unsigned)x; return m; }
template <class T> static inline T __shfl_sync(unsigned, T v, int src, int width = 32) {
    int l = (int)lane_id(); return exch(v, (l / width) * width + (src % width));
}
template <class T> static inline T __shfl_up_sync(unsigned, T v, unsigned d, int width = 32) {
    int l = (int)lane_id(); return exch(v, (l % width) >= (int)d ? l - (int)d : l);
}
template <class T> static inline T __shfl_down_sync(unsigned, T v, unsigned d, int width = 32) {
    int l = (int)lane_id(); return exch(v, (l % width) + (int)d < width ? l + (int)d : l);
}
static inline unsigned atomicAdd(unsigned *p, unsigned v) { return std::atomic_ref<unsigned>(*p).fetch_add(v); }
static inline unsigned atomicOr(unsigned *p, unsigned v) { return std::atomic_ref<unsigned>(*p).fetch_or(v); }
using std::min;
using std::max;
using std::log2;


#include "kernels.cuh"

template <class... P, class... A> static void emu_run(void (*k)(P...), unsigned grid, unsigned block, size_t smem, A... args) {
    if (block % 32) { fprintf(stderr, "emu: block size must be a multiple of 32\n"); abort(); }
    std::vector<unsigned char> dsm(smem + 16);
    for (unsigned b = 0; b < grid; b++) {
        std::barrier<> bb((std::ptrdiff_t)block);
        std::vector<Warp> warps(block / 32);
        std::vector<std::thread> th;
        for (unsigned t = 0; t < block; t++)
            th.emplace_back([&, t] {
                threadIdx.x = t; blockIdx.x = b; blockDim.x = block; emu_dsmem = dsm.data();
                emu_warp = &warps[t / 32]; emu_block_bar = &bb;
                k(((P)args)...);
            });
        for (auto &x : th) x.join();
    }
}

// What limits the GPU when the NPU runs. Each probe depends on one thing: alu and wmma on the shader clock, read_cached on the GPU cache,
// read_dram, write_dram and read_random on memory. Diagnostic only. gpu_probe [seconds]
#include <hip/hip_runtime.h>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

typedef float v8f __attribute__((ext_vector_type(8)));
typedef unsigned v4u __attribute__((ext_vector_type(4)));

__global__ void __launch_bounds__(256) alu_kernel(unsigned *out, int iterations) {
    unsigned v = threadIdx.x + 1;
    for (int i = 0; i < iterations; ++i) v = (v * 3u + 1u) ^ (v >> 3);
    out[blockIdx.x * 256 + threadIdx.x] = v;
}

__global__ void __launch_bounds__(256) wmma_kernel(float *out, int iterations) {
    __shared__ __align__(16) v4u a[16];
    a[threadIdx.x & 15] = v4u{0x3f803f80u + threadIdx.x, 0x3f803f80u, 0x3f803f80u, 0x3f803f80u};
    __syncthreads();
    v8f acc = {0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll 1
    for (int i = 0; i < iterations; ++i) {
        const v4u p = a[(i + threadIdx.x) & 15];
        union { v4u u[2]; v8f f; } r;
        r.u[0] = p;
        r.u[1] = p;
        acc = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(r.f, r.f, acc);
    }
    out[blockIdx.x * 256 + threadIdx.x] = acc[0];
}

// Each thread owns a region of steps * 16 bytes, so one launch touches thread count * steps * 16 bytes once. That is far more than the GPU caches hold.
__global__ void __launch_bounds__(256) read_kernel(unsigned *out, const v4u *in, size_t mask, int steps) {
    unsigned v = 0;
    const size_t base = size_t(blockIdx.x * 256 + threadIdx.x) * steps;
    for (int i = 0; i < steps; ++i) {
        const v4u w = in[(base + i) & mask];
        v += w.x + w.y + w.z + w.w;
    }
    out[blockIdx.x * 256 + threadIdx.x] = v;
}

__global__ void __launch_bounds__(256) write_kernel(v4u *out, size_t mask, int steps) {
    const size_t base = size_t(blockIdx.x * 256 + threadIdx.x) * steps;
    for (int i = 0; i < steps; ++i) out[(base + i) & mask] = v4u{unsigned(i), unsigned(i), unsigned(i), unsigned(i)};
}

// One line per step at a pseudo-random address: this depends on memory latency more than on bandwidth.
__global__ void __launch_bounds__(256) random_kernel(unsigned *out, const unsigned *in, size_t mask, int steps) {
    unsigned at = (blockIdx.x * 256 + threadIdx.x) * 2654435761u + 1u, v = 0;
    for (int i = 0; i < steps; ++i) {
        at = at * 1664525u + 1013904223u + v;
        v = in[(size_t(at) * 32) & mask] & 1u; // 1 for this buffer's contents only if the fill says so; the address depends on the last read.
    }
    out[blockIdx.x * 256 + threadIdx.x] = v;
}

int main(int argc, char **argv) {
    const double seconds = argc > 1 ? std::atof(argv[1]) : 8;
    hipStream_t s;
    hipStreamCreate(&s);
    const size_t big = size_t(1) << 28, cached = size_t(1) << 20;
    unsigned *a, *out;
    hipMalloc(&a, big);
    hipMalloc(&out, big);
    hipMemset(a, 1, big);
    const dim3 grid(128);
    struct Probe {
        std::string name;
        std::vector<double> ms;
    };
    std::vector<Probe> probes = {{"alu"}, {"wmma"}, {"read_cached"}, {"read_dram"}, {"write_dram"}, {"read_random"}};
    const size_t dram_bytes = size_t(128 * 256) * 400 * 16; // 210 MB per launch of the two DRAM probes.
    const auto end = std::chrono::steady_clock::now() + std::chrono::duration<double>(seconds);
    while (std::chrono::steady_clock::now() < end) {
        for (size_t p = 0; p < probes.size(); ++p) {
            const auto t0 = std::chrono::steady_clock::now();
            switch (p) {
                case 0: hipLaunchKernelGGL(alu_kernel, grid, dim3(256), 0, s, out, 12000); break;
                case 1: hipLaunchKernelGGL(wmma_kernel, grid, dim3(256), 0, s, reinterpret_cast<float *>(out), 2500); break;
                case 2: hipLaunchKernelGGL(read_kernel, grid, dim3(256), 0, s, out, reinterpret_cast<v4u *>(a), cached / 16 - 1, 400); break;
                case 3: hipLaunchKernelGGL(read_kernel, grid, dim3(256), 0, s, out, reinterpret_cast<v4u *>(a), big / 16 - 1, 400); break;
                case 4: hipLaunchKernelGGL(write_kernel, grid, dim3(256), 0, s, reinterpret_cast<v4u *>(out), big / 16 - 1, 400); break;
                case 5: hipLaunchKernelGGL(random_kernel, grid, dim3(256), 0, s, out, a, big / 4 - 1, 16); break;
            }
            hipStreamSynchronize(s);
            probes[p].ms.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
        }
    }
    for (auto &p : probes) {
        std::sort(p.ms.begin(), p.ms.end());
        printf("  %-12s median %7.3f ms  p10 %7.3f  p90 %7.3f  n=%zu", p.name.c_str(), p.ms[p.ms.size() / 2], p.ms[p.ms.size() / 10],
               p.ms[p.ms.size() * 9 / 10], p.ms.size());
        if (p.name == "read_dram" || p.name == "write_dram") printf("  %.1f GB/s", dram_bytes / (p.ms[p.ms.size() / 2] * 1e6));
        printf("\n");
    }
}

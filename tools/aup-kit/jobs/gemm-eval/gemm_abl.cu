// Which memory accesses of gemm_w4a16 cost time, and which of them the NPU slows down. Diagnostic only: results are not numerically useful.
// A copy of the kernel in which the weights (W), the input (X) or the output (O) touch only a small region, so that they stay in the GPU caches.
#include <hip/hip_runtime.h>
#include <drm/amdgpu_drm.h>
#include <drm/drm.h>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <unistd.h>
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <map>
#include <random>
#include <string>
#include <vector>

namespace {
__device__ __forceinline__ float bf16_to_float(uint16_t b) { return __uint_as_float(static_cast<uint32_t>(b) << 16); }
__device__ __forceinline__ uint16_t float_to_bf16(float f) {
    uint32_t u = __float_as_uint(f);
    u += 0x7fffu + ((u >> 16) & 1u);
    return static_cast<uint16_t>(u >> 16);
}
constexpr int64_t kTileK = 128, kTileN = 64, kTileBytes = 4352;
typedef float v4f __attribute__((ext_vector_type(4)));
typedef float v8f __attribute__((ext_vector_type(8)));
typedef uint32_t v4u __attribute__((ext_vector_type(4)));

constexpr int kFrag = 16;         // rows or columns per WMMA fragment
constexpr int kRowFrags = 4;      // fragments of A per wave
constexpr int kColFrags = 2;      // fragments of B per wave
constexpr int kBlockDim = 128;    // rows and columns per block
constexpr int kSumsPerTile = 16;  // floats per (fragment, K tile) in the row sums: 8 for half 0, then 8 for half 1
constexpr int kTileWordsX = 2048; // bf16 per (fragment, K tile) in the tile layout of the input

__device__ __forceinline__ v8f concat(v4f lo, v4f hi) { return __builtin_shufflevector(lo, hi, 0, 1, 2, 3, 4, 5, 6, 7); }

// Block (x, y) computes rows 128 * R * y .. + 128 * R - 1 and columns 128 * x .. + 127, with 8 * R waves. Each wave owns kRowFrags fragments
// of A and kColFrags fragments of B, so each wave runs 64 WMMA per K tile. The first 8 waves read the weights to LDS, once per K tile, and all waves use them.
template <int R, int O, int P>
__global__ void __launch_bounds__(256 * R + (P ? 32 : 0))
    gemm_w4a16_kernel(uint16_t *__restrict__ out, const uint16_t *__restrict__ xt, const float *__restrict__ sums,
                      const uint8_t *__restrict__ packed, int n, int tiles_k_used, int tiles_k, int tiles_n, int window_tiles, int xt_mask, int chunk_cols, int row_blocks) {
    const int wg_id = blockIdx.x;
    const int block_x_early = wg_id / (chunk_cols * row_blocks) * chunk_cols + wg_id % chunk_cols;
    if (P && threadIdx.x >= 256) {
        // A helper wave reads the next K tile while the other waves run WMMA, so that the page translations and cache lines are ready for them.
        const int lane = threadIdx.x - 256;
        int64_t bases[2];
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int ct = block_x_early * 2 + h;
            int64_t b = static_cast<int64_t>(ct / 8) * tiles_k;
#pragma unroll
            for (int g = 0; g < 8; ++g) {
                if (g < ct % 8) b += static_cast<int64_t>((tiles_n + 7 - g) / 8) * tiles_k;
            }
            bases[h] = b;
        }
        uint32_t got[5] = {0, 0, 0, 0, 0};
        for (int t = 0; t < tiles_k_used; ++t) {
            __syncthreads();
            if (t + 1 < tiles_k_used) {
#pragma unroll
                for (int i = 0; i < 5; ++i) {
                    const int line = (lane + 32 * i) % 136;
                    const uint8_t *tile = packed + ((bases[line / 68] + t + 1) % window_tiles) * kTileBytes;
                    got[i] = *reinterpret_cast<const uint32_t *>(tile + (line % 68) * 64);
                }
            }
            __syncthreads();
        }
        if ((got[0] | got[1] | got[2] | got[3] | got[4]) == 0x12345678u && lane == 99) out[0] = 0;
        return;
    }
    const int block_y = (wg_id / chunk_cols) % row_blocks;
    const int block_x = wg_id / (chunk_cols * row_blocks) * chunk_cols + wg_id % chunk_cols;
    __shared__ __align__(16) v4u bmat[kFrag][kBlockDim]; // [K row / 8][column]: bf16(128 + q) of 8 K rows, 2 per word
    __shared__ __align__(16) float tab_s[kBlockDim];               // scale per column, in the order that a lane reads
    __shared__ __align__(16) float tab_d[kBlockDim];               // 128 + zero per column, same order

    const int tid = threadIdx.x;
    const int lane = tid % 32;
    const int half = lane / 16;
    const int dim = lane % 16;
    const int wave = tid / 32;
    const int col_groups = 8 / kColFrags;
    const int wave_r = wave / col_groups;
    const int wave_c = wave % col_groups;

    // This thread's cell: tile tile_n of the block, K octet `octet`, 8 columns from cc * 8.
    const bool reads_weights = tid < 256;
    const int tile_n = (tid % 256) / 128;
    const int idx = tid % 128;
    const int octet = idx / 8;
    const int cc = idx % 8;
    const int col_tile = block_x * 2 + tile_n;
    int64_t base = static_cast<int64_t>(col_tile / 8) * tiles_k;
#pragma unroll
    for (int g = 0; g < 8; ++g) {
        if (g < col_tile % 8) {
            base += static_cast<int64_t>((tiles_n + 7 - g) / 8) * tiles_k;
        }
    }
    

    const int first_row = block_y * kBlockDim * R + wave_r * kRowFrags * kFrag;
    const uint16_t *row0[kRowFrags];
    const float *sums0[kRowFrags];
#pragma unroll
    for (int i = 0; i < kRowFrags; ++i) {
        const int xrow = ((xt_mask == 0) ? first_row % kBlockDim : first_row) / kFrag + i;
        row0[i] = xt + static_cast<int64_t>(xrow) * tiles_k_used * kTileWordsX + dim * 8;
        sums0[i] = sums + static_cast<int64_t>(xrow) * tiles_k_used * kSumsPerTile + half * 8;
    }

    v8f acc[kRowFrags][kColFrags];
#pragma unroll
    for (int i = 0; i < kRowFrags; ++i) {
#pragma unroll
        for (int j = 0; j < kColFrags; ++j) {
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                acc[i][j][e] = 0.0f;
            }
        }
    }
    float scale[kColFrags], offset[kColFrags]; // of the tile in acc

    for (int t = 0; t < tiles_k_used; ++t) {
        const uint8_t *tile = packed + ((base + t) % window_tiles) * kTileBytes;

        // 1. Weights to LDS. The nibble of a word at bit 4 * c is column c of the cell.
        if (reads_weights) {
            const uint8_t *cell = tile + octet * 256 + cc * 32;
            const v4u w0 = *reinterpret_cast<const v4u *>(cell);
            const v4u w1 = *reinterpret_cast<const v4u *>(cell + 16);
            const uint32_t rows[8] = {w0.x, w0.y, w0.z, w0.w, w1.x, w1.y, w1.z, w1.w};
            uint32_t col_words[8][4];
#pragma unroll
            for (int p = 0; p < 4; ++p) {
                const uint32_t lo = (rows[2 * p] & 0xFFFFu) | (rows[2 * p + 1] << 16);
                const uint32_t hi = (rows[2 * p] >> 16) | (rows[2 * p + 1] & 0xFFFF0000u);
#pragma unroll
                for (int c = 0; c < 4; ++c) {
                    col_words[c][p] = ((lo >> (4 * c)) & 0x000F000Fu) | 0x43004300u;
                    col_words[c + 4][p] = ((hi >> (4 * c)) & 0x000F000Fu) | 0x43004300u;
                }
            }
            const int col0 = tile_n * kTileN + cc * 8;
#pragma unroll
            for (int c = 0; c < 8; ++c) {
                const v4u v = {col_words[c][0], col_words[c][1], col_words[c][2], col_words[c][3]};
                bmat[octet][col0 + c] = v;
            }
            if (idx < kTileN) {
                const int col = tile_n * kTileN + idx;
                // A zero scale would make the division below infinite. With this floor, such a tile adds about nothing.
                const float s = fmaxf(bf16_to_float(*reinterpret_cast<const uint16_t *>(tile + 4096 + 2 * idx)), 1e-20f);
                const int z = tile[4224 + (idx / 8) * 16 + idx % 8];
                const int slot = (col / (kFrag * kColFrags) * kFrag + col % kFrag) * kColFrags + (col % (kFrag * kColFrags)) / kFrag;
                tab_s[slot] = s;
                tab_d[slot] = static_cast<float>(128 + z);
            }
        }
        __syncthreads();

        // 2. Finish the previous tile, and change acc to the units of this tile.
        float new_scale[kColFrags];
#pragma unroll
        for (int j = 0; j < kColFrags; ++j) {
            new_scale[j] = tab_s[(wave_c * kFrag + dim) * kColFrags + j];
        }
        if (t > 0) {
#pragma unroll
            for (int j = 0; j < kColFrags; ++j) {
                scale[j] /= new_scale[j];
            }
#pragma unroll
            for (int i = 0; i < kRowFrags; ++i) {
                const float *sp = sums0[i] + static_cast<int64_t>((t - 1) & xt_mask) * kSumsPerTile;
                const v8f xs = concat(*reinterpret_cast<const v4f *>(sp), *reinterpret_cast<const v4f *>(sp + 4));
#pragma unroll
                for (int j = 0; j < kColFrags; ++j) {
#pragma unroll
                    for (int e = 0; e < 8; ++e) {
                        acc[i][j][e] = scale[j] * __builtin_fmaf(-offset[j], xs[e], acc[i][j][e]);
                    }
                }
            }
        }
#pragma unroll
        for (int j = 0; j < kColFrags; ++j) {
            scale[j] = new_scale[j];
            offset[j] = tab_d[(wave_c * kFrag + dim) * kColFrags + j];
        }

        // 3. WMMA.
        const uint16_t *tile_row[kRowFrags];
#pragma unroll
        for (int i = 0; i < kRowFrags; ++i) {
            tile_row[i] = row0[i] + static_cast<int64_t>(t & xt_mask) * kTileWordsX;
        }
#pragma unroll
        for (int ki = 0; ki < 8; ++ki) {
            v8f b[kColFrags];
#pragma unroll
            for (int j = 0; j < kColFrags; ++j) {
                const int col = wave_c * kColFrags * kFrag + j * kFrag + dim;
                b[j] = concat(*reinterpret_cast<const v4f *>(&bmat[ki * 2][col]), *reinterpret_cast<const v4f *>(&bmat[ki * 2 + 1][col]));
            }
#pragma unroll
            for (int i = 0; i < kRowFrags; ++i) {
                const uint16_t *ap = tile_row[i] + ki * 2 * kFrag * 8;
                const v8f a = concat(*reinterpret_cast<const v4f *>(ap), *reinterpret_cast<const v4f *>(ap + kFrag * 8));
#pragma unroll
                for (int j = 0; j < kColFrags; ++j) {
                    acc[i][j] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a, b[j], acc[i][j]);
                }
            }
        }
        __syncthreads();
    }

    // 4. Finish the last tile, and write the output.
#pragma unroll
    for (int i = 0; i < kRowFrags; ++i) {
        const float *sp = sums0[i] + static_cast<int64_t>((tiles_k_used - 1) & xt_mask) * kSumsPerTile;
        const v8f xs = concat(*reinterpret_cast<const v4f *>(sp), *reinterpret_cast<const v4f *>(sp + 4));
#pragma unroll
        for (int j = 0; j < kColFrags; ++j) {
            const int col = block_x * kBlockDim + wave_c * kColFrags * kFrag + j * kFrag + dim;
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                const int row = first_row + i * kFrag + 2 * e + half;
                out[O ? (static_cast<int64_t>(row % kBlockDim) * kBlockDim + col % kBlockDim) : static_cast<int64_t>(row) * n + col] = float_to_bf16(scale[j] * __builtin_fmaf(-offset[j], xs[e], acc[i][j][e]));
            }
        }
    }
}


typedef uint32_t v4u_copy __attribute__((ext_vector_type(4)));
__global__ void copy_kernel(v4u_copy *d, const v4u_copy *s) {
    const size_t i = size_t(blockIdx.x) * 256 + threadIdx.x;
    d[i] = s[i];
}

long counter(const char *name) {
    long v = 0;
    std::ifstream f(std::string("/sys/class/drm/card0/device/") + name);
    f >> v;
    return v;
}

// VRAM through the render node: KFD gives this APU only GTT for every HIP and HSA pool.
void *vram_alloc(size_t bytes, bool uswc = false) {
    static const int fd = open("/dev/dri/renderD128", O_RDWR | O_CLOEXEC);
    union drm_amdgpu_gem_create args = {};
    args.in.bo_size = bytes;
    args.in.alignment = size_t(2) << 20;
    args.in.domains = uswc ? AMDGPU_GEM_DOMAIN_GTT : AMDGPU_GEM_DOMAIN_VRAM;
    args.in.domain_flags = uswc ? AMDGPU_GEM_CREATE_CPU_GTT_USWC : 0;
    const long v0 = counter("mem_info_vram_used"), g0 = counter("mem_info_gtt_used");
    if (fd < 0 || ioctl(fd, DRM_IOCTL_AMDGPU_GEM_CREATE, &args) != 0) { perror("GEM_CREATE"); std::exit(1); }
    printf("    GEM_CREATE %.1f MB: VRAM +%.1f GTT +%.1f\n", bytes / 1048576.0, (counter("mem_info_vram_used") - v0) / 1048576.0, (counter("mem_info_gtt_used") - g0) / 1048576.0);
    struct drm_prime_handle prime = {};
    prime.handle = args.out.handle;
    prime.flags = DRM_CLOEXEC | DRM_RDWR;
    if (ioctl(fd, DRM_IOCTL_PRIME_HANDLE_TO_FD, &prime) != 0) { perror("PRIME"); std::exit(1); }
    hipExternalMemoryHandleDesc desc = {};
    desc.type = hipExternalMemoryHandleTypeOpaqueFd;
    desc.handle.fd = prime.fd;
    desc.size = bytes;
    hipExternalMemory_t mem;
    hipError_t e = hipImportExternalMemory(&mem, &desc);
    if (e != hipSuccess) { printf("hipImportExternalMemory failed: %d\n", e); std::exit(1); }
    printf("    import: VRAM +%.1f GTT +%.1f\n", (counter("mem_info_vram_used") - v0) / 1048576.0, (counter("mem_info_gtt_used") - g0) / 1048576.0);
    hipExternalMemoryBufferDesc bd = {};
    bd.size = bytes;
    void *p = nullptr;
    e = hipExternalMemoryGetMappedBuffer(&p, mem, &bd);
    if (e != hipSuccess) { printf("hipExternalMemoryGetMappedBuffer failed: %d\n", e); std::exit(1); }
    printf("    mapped: VRAM +%.1f GTT +%.1f\n", (counter("mem_info_vram_used") - v0) / 1048576.0, (counter("mem_info_gtt_used") - g0) / 1048576.0);
    return p;
}

} // namespace

int main(int argc, char **argv) {
    const int rounds = argc > 1 ? std::atoi(argv[1]) : 20;
    const int m = 512, k = 4096, n = 15360, tk = k / 128, tn = n / 64;
    hipStream_t s;
    hipStreamCreate(&s);
    const size_t packed_bytes = size_t(tk) * tn * kTileBytes, xt_bytes = size_t(m) * k * 2, sum_bytes = size_t(m / 16) * tk * 16 * 4, out_bytes = size_t(m) * n * 2;
    uint8_t *packed;
    uint16_t *xt, *out;
    float *sums;
    hipMalloc(&packed, packed_bytes);
    hipMalloc(&xt, xt_bytes);
    hipMalloc(&sums, sum_bytes);
    hipMalloc(&out, out_bytes);
    std::mt19937_64 rng(1);
    std::vector<uint8_t> hp(packed_bytes);
    for (size_t i = 0; i < packed_bytes; i += 8) { const uint64_t v = rng(); std::copy_n(reinterpret_cast<const uint8_t *>(&v), std::min<size_t>(8, packed_bytes - i), &hp[i]); }
    hipMemcpy(packed, hp.data(), packed_bytes, hipMemcpyHostToDevice);
    std::vector<uint16_t> hx(xt_bytes / 2);
    for (auto &v : hx) v = 0x3f00 + (rng() & 0xff);
    hipMemcpy(xt, hx.data(), xt_bytes, hipMemcpyHostToDevice);
    std::vector<float> hs(sum_bytes / 4, 1.0f);
    hipMemcpy(sums, hs.data(), sum_bytes, hipMemcpyHostToDevice);
    const int all_tiles = tk * tn;
    const long vram0 = counter("mem_info_vram_used"), gtt0 = counter("mem_info_gtt_used");
    const auto copy_to = [&](void *d, const void *src, size_t bytes) {
        hipLaunchKernelGGL(copy_kernel, dim3((bytes + 4095) / 4096), dim3(256), 0, s, reinterpret_cast<v4u_copy *>(d), reinterpret_cast<const v4u_copy *>(src));
        hipStreamSynchronize(s);
        if (hipGetLastError() != hipSuccess) { printf("copy failed\n"); std::exit(1); }
    };
    const size_t huge = size_t(2) << 20;
    const auto rounded = [&](size_t b) { return (b + huge - 1) / huge * huge; };
    const auto uswc = [&](const void *src, size_t bytes) { void *d = vram_alloc(rounded(bytes), true); copy_to(d, src, bytes); return d; };
    auto *wpacked = static_cast<uint8_t *>(uswc(packed, packed_bytes));
    auto *wxt = static_cast<uint16_t *>(uswc(xt, xt_bytes));
    auto *wsums = static_cast<float *>(uswc(sums, sum_bytes));
    auto *wout = static_cast<uint16_t *>(uswc(out, out_bytes));
    printf("  VRAM +%.1f MB, GTT +%.1f MB for the USWC copies\n", (counter("mem_info_vram_used") - vram0) / 1048576.0, (counter("mem_info_gtt_used") - gtt0) / 1048576.0);
    void *host = nullptr, *uncached = nullptr, *fine = nullptr;
    hipHostMalloc(&host, rounded(packed_bytes), hipHostMallocNonCoherent);
    copy_to(host, packed, packed_bytes);
    hipExtMallocWithFlags(&uncached, rounded(packed_bytes), hipDeviceMallocUncached);
    copy_to(uncached, packed, packed_bytes);
    hipExtMallocWithFlags(&fine, rounded(packed_bytes), hipDeviceMallocFinegrained);
    copy_to(fine, packed, packed_bytes);
    struct Case {
        std::string name;
        const uint8_t *packed;
        const uint16_t *xt;
        const float *sums;
        uint16_t *out;
        int window;
    };
    const std::vector<Case> cases = {
        {"default", packed, xt, sums, out, all_tiles},
        {"default_W64k", packed, xt, sums, out, int(0.0625 * 1048576 / kTileBytes)},
        {"uswc_weights", wpacked, xt, sums, out, all_tiles},
        {"uswc_all", wpacked, wxt, wsums, wout, all_tiles},
        {"host_weights", static_cast<uint8_t *>(host), xt, sums, out, all_tiles},
        {"uncached_weights", static_cast<uint8_t *>(uncached), xt, sums, out, all_tiles},
        {"fine_weights", static_cast<uint8_t *>(fine), xt, sums, out, all_tiles},
    };
    const int col_blocks = n / 128, row_blocks = m / 128;
    std::map<std::string, std::vector<double>> ms;
    const auto run = [&](const Case &c) {
        hipLaunchKernelGGL((gemm_w4a16_kernel<1, 0, 0>), dim3(col_blocks * row_blocks), dim3(256), 0, s, c.out, c.xt, c.sums, c.packed, n, tk, tk, tn, c.window, ~0, 120, row_blocks);
    };
    for (int r = 0; r < rounds; ++r) {
        for (const auto &c : cases) {
            for (int i = 0; i < 3; ++i) run(c);
            hipStreamSynchronize(s);
            const auto t0 = std::chrono::steady_clock::now();
            for (int i = 0; i < 10; ++i) run(c);
            hipStreamSynchronize(s);
            ms[c.name].push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / 10);
        }
    }
    for (const auto &c : cases) {
        auto &v = ms[c.name];
        std::sort(v.begin(), v.end());
        printf("  %-18s median %7.3f ms  p10 %7.3f  p90 %7.3f\n", c.name.c_str(), v[v.size() / 2], v[v.size() / 10], v[v.size() * 9 / 10]);
    }
}

// W4A16 GEMM (M > 1) on the HeteroMosaic packed layout. See gemm_w4a16() in gpu_ffn.hpp.
//
// v_wmma alone reaches 20.5 TFLOPS on the HX 370 (measured, 33 cycles per WMMA per SIMD), hipkernels::w4a16_gemm_fused_packed
// reaches 9.3 and this kernel 12.4. Differences:
//   - WMMA gets exact integer weights. bf16(128 + q) has the bits 0x4300 | q, so unpacking needs no conversion or rounding.
//     Zero and scale are applied in fp32, once per tile and element. Per tile, with P = sum_k x_k * (128 + q_k) from WMMA,
//     sum_k x_k * (q_k - z) * s = s * (P - d * X), where d = 128 + z and X = sum_k x_k comes from a first pass.
//   - the WMMA accumulator is also the result, in units of the current tile's scale: C += P during a tile, and between tiles
//     C = (C - d * X) * s_old / s_new. A separate result register per element would take 64 more VGPRs, and the kernel
//     then spills to scratch and runs 13 to 25 times slower (measured).
//   - each thread unpacks one 32-byte cell of 8 rows and 8 columns.
//   - neighbouring lanes read neighbouring addresses. A stride of 256 bytes between lanes in LDS makes the kernel 1.3 times
//     slower (measured), and the row stride of the input makes it 1.14 times slower (measured), so the first pass also copies the
//     input to a tile layout. A wave owns 4 fragments of A and 2 of B. The shape 2x4 takes about the same time, and 8x1 takes 1.1 times as long (measured).
//
// Measured but not useful: loading A only in the 16 lanes that WMMA reads, one wave per workgroup without LDS, and
// delaying half of the waves. Without barriers the kernel is slower too.
//
// WMMA registers on gfx11, wave32 (measured). Each lane has 8 dwords of A, 8 of B and 8 of C.
//   half = lane / 16, dim = lane % 16
//   A: 16 bf16 of row dim. WMMA reads a lane only if half == dim % 2.
//   B: 16 bf16 of column dim. WMMA reads all lanes.
//   C: fp32 C[row 2 * e + half, column dim] in element e = 0..7.
//   A and B must hold their K values in the same order. rocWMMA swaps the order in half 1, which is not needed here.
//
// Tile layout: see pack_w4a16 and gemv_w4a16.cu.
#include "bf16_device.hpp"
#include "heteromega/gpu_ffn.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <mutex>
#include <unordered_map>
#include <utility>

namespace heteromega {
namespace {

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

// Input in the order that the WMMA loop reads it, so that neighbouring lanes read one cache line.
// xt[(row / 16 * tiles + t) * 2048 + a * 128 + (row % 16) * 8 ..] = x[row, 128 * t + 8 * a .. + 7].
// sums[(row / 16 * tiles + t) * 16 + (row & 1) * 8 + (row & 15) / 2] = sum of x[row, 128 * t .. 128 * t + 127].
// So a lane of half h reads the sums of its 8 rows of C in one place.
__global__ void __launch_bounds__(32)
    prepare_input_kernel(const uint16_t *__restrict__ x, float *__restrict__ sums, uint16_t *__restrict__ xt, int k, int tiles) {
    const int t = blockIdx.x;
    const int row = blockIdx.y * 32 + threadIdx.x;
    const uint16_t *p = x + static_cast<int64_t>(row) * k + static_cast<int64_t>(t) * kTileK;
    uint16_t *q = xt + (static_cast<int64_t>(row / kFrag) * tiles + t) * kTileWordsX + (row % kFrag) * 8;
    float sum = 0.0f;
#pragma unroll
    for (int a = 0; a < kFrag; ++a) {
        const v4u v = *reinterpret_cast<const v4u *>(p + 8 * a);
        *reinterpret_cast<v4u *>(q + a * kFrag * 8) = v;
#pragma unroll
        for (int w = 0; w < 4; ++w) {
            sum += bf16_to_float(static_cast<uint16_t>(v[w])) + __uint_as_float(v[w] & 0xFFFF0000u);
        }
    }
    sums[(static_cast<int64_t>(row / kFrag) * tiles + t) * kSumsPerTile + (row & 1) * 8 + (row & 15) / 2] = sum;
}

// Block (x, y) computes rows 128 * y .. + 127 and columns 128 * x .. + 127, with 8 waves. Each wave owns kRowFrags fragments of A and kColFrags
// fragments of B, so each wave runs 64 WMMA per K tile.
// With Fuse, the columns of a block are a tile of gate and the tile of up of the same 64 channels (GateUpLayout::kInterleaved). The kernel then writes
// SiLU(gate) * up to `out` in the tile layout of prepare_input_kernel, and half_sums per half of a K tile. So gate and up never go to memory.
template <bool Fuse>
__global__ void __launch_bounds__(256)
    gemm_w4a16_kernel(uint16_t *__restrict__ out, const uint16_t *__restrict__ xt, const float *__restrict__ sums,
                      const uint8_t *__restrict__ packed, int n, int tiles_k_used, int tiles_k, int tiles_n, float *__restrict__ half_sums) {
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
    const int tile_n = tid / 128;
    const int idx = tid % 128;
    const int octet = idx / 8;
    const int cc = idx % 8;
    const int col_tile = blockIdx.x * 2 + tile_n;
    int64_t base = static_cast<int64_t>(col_tile / 8) * tiles_k;
#pragma unroll
    for (int g = 0; g < 8; ++g) {
        if (g < col_tile % 8) {
            base += static_cast<int64_t>((tiles_n + 7 - g) / 8) * tiles_k;
        }
    }
    const uint8_t *tile0 = packed + base * kTileBytes;

    const int first_row = blockIdx.y * kBlockDim + wave_r * kRowFrags * kFrag;
    const uint16_t *row0[kRowFrags];
    const float *sums0[kRowFrags];
#pragma unroll
    for (int i = 0; i < kRowFrags; ++i) {
        row0[i] = xt + static_cast<int64_t>(first_row / kFrag + i) * tiles_k_used * kTileWordsX + dim * 8;
        sums0[i] = sums + static_cast<int64_t>(first_row / kFrag + i) * tiles_k_used * kSumsPerTile + half * 8;
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
        const uint8_t *tile = tile0 + static_cast<int64_t>(t) * kTileBytes;

        // 1. Weights to LDS. The nibble of a word at bit 4 * c is column c of the cell.
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
                const float *sp = sums0[i] + static_cast<int64_t>(t - 1) * kSumsPerTile;
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
            tile_row[i] = row0[i] + static_cast<int64_t>(t) * kTileWordsX;
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
    if (Fuse) {
        // The weights are done, so bmat holds [row of the block][channel of the block] now: SiLU(gate) from the gate waves, then the product from the up waves.
        auto *lds = reinterpret_cast<float (*)[kTileN]>(&bmat[0][0]);
#pragma unroll
        for (int pass = 0; pass < 2; ++pass) {
            if (wave_c / 2 == pass) {
#pragma unroll
                for (int i = 0; i < kRowFrags; ++i) {
                    const float *sp = sums0[i] + static_cast<int64_t>(tiles_k_used - 1) * kSumsPerTile;
                    const v8f xs = concat(*reinterpret_cast<const v4f *>(sp), *reinterpret_cast<const v4f *>(sp + 4));
#pragma unroll
                    for (int j = 0; j < kColFrags; ++j) {
                        const int channel = (wave_c % 2) * kColFrags * kFrag + j * kFrag + dim;
#pragma unroll
                        for (int e = 0; e < 8; ++e) {
                            const int row = wave_r * kRowFrags * kFrag + i * kFrag + 2 * e + half;
                            const float v = scale[j] * __builtin_fmaf(-offset[j], xs[e], acc[i][j][e]);
                            if (pass == 0) {
                                lds[row][channel] = v / (1.0f + __expf(-v));
                            } else {
                                lds[row][channel] *= v;
                            }
                        }
                    }
                }
            }
            __syncthreads();
        }
        if (tid < kBlockDim) {
            const int row = blockIdx.y * kBlockDim + tid;
            const int tile = blockIdx.x / 2, second = blockIdx.x % 2;
            const int64_t place = static_cast<int64_t>(row / kFrag) * (n / (2 * kBlockDim)) + tile;
            uint16_t *q = out + place * kTileWordsX + second * 8 * kFrag * 8 + (row % kFrag) * 8;
            float sum = 0.0f; // Of the rounded values, because WMMA of the next GEMM reads those.
#pragma unroll
            for (int a = 0; a < 8; ++a) {
                v4u v;
#pragma unroll
                for (int w = 0; w < 4; ++w) {
                    const uint16_t lo = float_to_bf16(lds[tid][a * 8 + 2 * w]), hi = float_to_bf16(lds[tid][a * 8 + 2 * w + 1]);
                    sum += bf16_to_float(lo) + bf16_to_float(hi);
                    v[w] = static_cast<uint32_t>(lo) | (static_cast<uint32_t>(hi) << 16);
                }
                *reinterpret_cast<v4u *>(q + a * kFrag * 8) = v;
            }
            half_sums[place * 2 * kSumsPerTile + second * kSumsPerTile + (row & 1) * 8 + (row & 15) / 2] = sum;
        }
    } else {
#pragma unroll
        for (int i = 0; i < kRowFrags; ++i) {
            const float *sp = sums0[i] + static_cast<int64_t>(tiles_k_used - 1) * kSumsPerTile;
            const v8f xs = concat(*reinterpret_cast<const v4f *>(sp), *reinterpret_cast<const v4f *>(sp + 4));
#pragma unroll
            for (int j = 0; j < kColFrags; ++j) {
                const int col = blockIdx.x * kBlockDim + wave_c * kColFrags * kFrag + j * kFrag + dim;
#pragma unroll
                for (int e = 0; e < 8; ++e) {
                    const int row = first_row + i * kFrag + 2 * e + half;
                    out[static_cast<int64_t>(row) * n + col] = float_to_bf16(scale[j] * __builtin_fmaf(-offset[j], xs[e], acc[i][j][e]));
                }
            }
        }
    }
}

// sums of a K tile = the two half_sums of the Fuse kernel.
__global__ void __launch_bounds__(128) add_half_sums_kernel(const float *__restrict__ half, float *__restrict__ sums) {
    const int64_t i = static_cast<int64_t>(blockIdx.x) * 128 + threadIdx.x;
    const float *p = half + i / kSumsPerTile * 2 * kSumsPerTile + i % kSumsPerTile;
    sums[i] = p[0] + p[kSumsPerTile];
}

__global__ void copy_words_kernel(v4u *__restrict__ d, const v4u *__restrict__ s) {
    const size_t i = static_cast<size_t>(blockIdx.x) * 256 + threadIdx.x;
    d[i] = s[i];
}

// With HETEROMEGA_GEMM_UNCACHED=1, the kernel reads a copy of the weights from uncached memory. Then the weights do not push the input out of
// the GPU L2. Without the NPU this changes nothing; with the NPU in a loop it saves 6% to 11% (measured). A process has one copy per matrix, made at first
// use, so the weights must not change or move while the process runs.
bool uncached_weights_enabled() {
    static const bool enabled = [] {
        const char *v = std::getenv("HETEROMEGA_GEMM_UNCACHED");
        return v != nullptr && std::atoi(v) != 0;
    }();
    return enabled;
}

std::mutex uncached_mutex;
std::map<std::pair<const uint8_t *, size_t>, const uint8_t *> uncached_copies;

// Call with uncached_mutex held. Null means no memory. A matrix without a copy reads its normal weights: the output stays the same and only the
// speed changes, so the run goes on and the log says so.
uint8_t *make_uncached_copy(const uint8_t *packed, size_t bytes) {
    static hipStream_t copy_stream = nullptr;
    if (copy_stream == nullptr) {
        TORCH_CHECK(hipStreamCreate(&copy_stream) == hipSuccess, "gemm_w4a16: cannot create the stream for weight copies");
    }
    uint8_t *copy = nullptr;
    const size_t rounded = (bytes + 4095) / 4096 * 4096;
    const hipError_t allocated = hipExtMallocWithFlags(reinterpret_cast<void **>(&copy), rounded, hipDeviceMallocUncached);
    if (allocated != hipSuccess) {
        std::fprintf(stderr, "gemm_w4a16: UNCACHED_FALLBACK %zu bytes: %s\n", rounded, hipGetErrorString(allocated));
        return nullptr;
    }
    hipLaunchKernelGGL(copy_words_kernel, dim3(static_cast<unsigned>(rounded / 4096)), dim3(256), 0, copy_stream, reinterpret_cast<v4u *>(copy),
                       reinterpret_cast<const v4u *>(packed));
    check_launch("gemm_w4a16 weight copy");
    TORCH_CHECK(hipStreamSynchronize(copy_stream) == hipSuccess, "gemm_w4a16: weight copy failed");
    return copy;
}

const uint8_t *weights_to_read(const uint8_t *packed, size_t bytes) {
    if (!uncached_weights_enabled()) {
        return packed;
    }
    std::lock_guard<std::mutex> lock(uncached_mutex);
    const auto key = std::make_pair(packed, bytes);
    auto it = uncached_copies.find(key);
    if (it != uncached_copies.end()) {
        return it->second;
    }
    const uint8_t *copy = make_uncached_copy(packed, bytes);
    if (copy == nullptr) {
        copy = packed;
    }
    uncached_copies.emplace(key, copy);
    return copy;
}

struct InputScratch {
    float *data = nullptr;
    size_t bytes = 0;
};

// Scratch for the row sums and the input in tile layout, one per stream. Kernels of a stream run in order, so its next GEMM can reuse it.
// A buffer that becomes too small stays allocated, because kernels on the stream can still use it. A process has few sizes.
float *input_scratch(size_t bytes, hipStream_t stream, int which = 0) {
    static std::mutex mutex;
    static std::unordered_map<hipStream_t, InputScratch> buffers[2];
    std::lock_guard<std::mutex> lock(mutex);
    InputScratch &b = buffers[which][stream];
    if (b.bytes < bytes) {
        TORCH_CHECK(hipMalloc(&b.data, bytes) == hipSuccess, "gemm_w4a16: cannot allocate ", bytes, " bytes");
        b.bytes = bytes;
    }
    return b.data;
}

struct InputReuse {
    bool active = false;
    const void *x = nullptr;
    int64_t k = 0;
    hipStream_t stream = nullptr;
    int hits = 0;
};
thread_local InputReuse input_reuse;

} // namespace

torch::Tensor gemm_uncached_weights(const torch::Tensor &packed) {
    if (!uncached_weights_enabled()) {
        return packed;
    }
    std::lock_guard<std::mutex> lock(uncached_mutex);
    const size_t bytes = static_cast<size_t>(packed.numel());
    const uint8_t *copy = make_uncached_copy(packed.data_ptr<uint8_t>(), bytes);
    if (copy == nullptr) {
        return packed;
    }
    // The key is the copy's own address, not the source's: the allocator gives the source's address to the next matrix.
    uncached_copies.emplace(std::make_pair(copy, bytes), copy);
    return torch::from_blob(const_cast<uint8_t *>(copy), {packed.numel()}, [](void *) {}, torch::TensorOptions().dtype(torch::kUInt8).device(packed.device()));
}

GemmInputReuse::GemmInputReuse() { input_reuse = InputReuse{true}; }
GemmInputReuse::~GemmInputReuse() { input_reuse = InputReuse{}; }
void GemmInputReuse::mark_prepared(const void *x, int64_t k, hipStream_t stream) {
    input_reuse.x = x;
    input_reuse.k = k;
    input_reuse.stream = stream;
}
int GemmInputReuse::hits() const { return input_reuse.hits; }

void gemm_input_buffers(int64_t m, int64_t k, hipStream_t stream, float **sums, uint16_t **tiles) {
    const size_t sum_bytes = static_cast<size_t>(m / kFrag) * (k / kTileK) * kSumsPerTile * sizeof(float);
    const size_t offset = (sum_bytes + 4095) / 4096 * 4096;
    *sums = input_scratch(offset + static_cast<size_t>(m) * k * sizeof(uint16_t), stream);
    *tiles = reinterpret_cast<uint16_t *>(reinterpret_cast<char *>(*sums) + offset);
}

bool gemm_w4a16_supports(int64_t m, int64_t k, int64_t n) { return m > 0 && m % kTileK == 0 && k % kTileK == 0 && n % kTileK == 0; }

void gemm_w4a16(torch::Tensor &out, const torch::Tensor &x, const torch::Tensor &packed, int64_t k, int64_t n, int64_t packed_k,
                int64_t packed_n, hipStream_t stream) {
    const int64_t m = x.numel() / k;
    TORCH_CHECK(gemm_w4a16_supports(m, k, n), "gemm_w4a16: M, k and n must be multiples of 128, got ", m, ", ", k, ", ", n);
    TORCH_CHECK(k <= packed_k && n <= packed_n && packed_k % kTileK == 0 && packed_n % kTileN == 0, "gemm_w4a16: packed matrix [",
                packed_k, ", ", packed_n, "] does not contain [", k, ", ", n, "]");
    TORCH_CHECK(packed.numel() == (packed_k / kTileK) * (packed_n / kTileN) * kTileBytes, "gemm_w4a16: packed size does not match ",
                packed_k, "x", packed_n);
    TORCH_CHECK(x.is_contiguous() && x.scalar_type() == torch::kBFloat16, "gemm_w4a16: x must be contiguous bf16");
    TORCH_CHECK(out.numel() == m * n && out.is_contiguous() && out.scalar_type() == torch::kBFloat16, "gemm_w4a16: out must be contiguous bf16 [",
                m, ", ", n, "]");

    const int tiles_used = static_cast<int>(k / kTileK);
    const int tk = static_cast<int>(packed_k / kTileK), tn = static_cast<int>(packed_n / kTileN);
    const auto *xp = reinterpret_cast<const uint16_t *>(x.data_ptr());
    auto *yp = reinterpret_cast<uint16_t *>(out.data_ptr());
    const auto *pk = weights_to_read(packed.data_ptr<uint8_t>(), packed.numel());

    float *sums = nullptr;
    uint16_t *xt = nullptr;
    gemm_input_buffers(m, k, stream, &sums, &xt);
    if (input_reuse.active && input_reuse.x == xp && input_reuse.k == k && input_reuse.stream == stream) {
        ++input_reuse.hits;
    } else {
        hipLaunchKernelGGL(prepare_input_kernel, dim3(tiles_used, static_cast<unsigned>(m / 32)), dim3(32), 0, stream, xp, sums, xt,
                           static_cast<int>(k), tiles_used);
        check_launch("gemm_w4a16 input");
        if (input_reuse.active) {
            input_reuse.x = xp;
            input_reuse.k = k;
            input_reuse.stream = stream;
        }
    }

    hipLaunchKernelGGL(gemm_w4a16_kernel<false>, dim3(static_cast<unsigned>(n / kBlockDim), static_cast<unsigned>(m / kBlockDim)), dim3(256), 0, stream, yp, xt, sums, pk,
                       static_cast<int>(n), tiles_used, tk, tn, static_cast<float *>(nullptr));
    check_launch("gemm_w4a16");
}

bool gemm_w4a16_ffn_supports(int64_t m, int64_t h, int64_t n) { return m > 0 && m % kTileK == 0 && h % kTileK == 0 && n % kTileK == 0; }

void gemm_w4a16_ffn(torch::Tensor &y, const torch::Tensor &x, const torch::Tensor &gate_up, const torch::Tensor &down, int64_t h, int64_t n, int64_t stored, hipStream_t stream) {
    const int64_t m = x.numel() / h;
    TORCH_CHECK(gemm_w4a16_ffn_supports(m, h, n), "gemm_w4a16_ffn: M, H and n must be multiples of 128, got ", m, ", ", h, ", ", n);
    TORCH_CHECK(n <= stored && stored % kTileK == 0, "gemm_w4a16_ffn: packed channels ", stored, " do not contain ", n);
    TORCH_CHECK(gate_up.numel() == (h / kTileK) * (2 * stored / kTileN) * kTileBytes && down.numel() == (stored / kTileK) * (h / kTileN) * kTileBytes,
                "gemm_w4a16_ffn: packed sizes do not match ", h, "x", stored);
    TORCH_CHECK(x.is_contiguous() && x.scalar_type() == torch::kBFloat16 && y.numel() == m * h && y.is_contiguous() && y.scalar_type() == torch::kBFloat16,
                "gemm_w4a16_ffn: x and y must be contiguous bf16 [", m, ", ", h, "]");
    const auto *xp = reinterpret_cast<const uint16_t *>(x.data_ptr());
    auto *yp = reinterpret_cast<uint16_t *>(y.data_ptr());
    const auto *gate_up_weights = weights_to_read(gate_up.data_ptr<uint8_t>(), gate_up.numel());
    const auto *down_weights = weights_to_read(down.data_ptr<uint8_t>(), down.numel());

    const int tiles_h = static_cast<int>(h / kTileK), tiles_act = static_cast<int>(n / kTileK);
    const auto round_up = [](size_t bytes) { return (bytes + 4095) / 4096 * 4096; };

    const size_t x_sum_bytes = static_cast<size_t>(m / kFrag) * tiles_h * kSumsPerTile * sizeof(float);
    float *x_sums = input_scratch(round_up(x_sum_bytes) + static_cast<size_t>(m) * h * sizeof(uint16_t), stream);
    auto *x_tiles = reinterpret_cast<uint16_t *>(reinterpret_cast<char *>(x_sums) + round_up(x_sum_bytes));

    const size_t act_sum_bytes = static_cast<size_t>(m / kFrag) * tiles_act * kSumsPerTile * sizeof(float);
    float *act_half_sums = input_scratch(round_up(3 * act_sum_bytes) + static_cast<size_t>(m) * n * sizeof(uint16_t), stream, 1);
    float *act_sums = act_half_sums + 2 * act_sum_bytes / sizeof(float);
    auto *act_tiles = reinterpret_cast<uint16_t *>(reinterpret_cast<char *>(act_half_sums) + round_up(3 * act_sum_bytes));

    if (input_reuse.active && input_reuse.x == xp && input_reuse.k == h && input_reuse.stream == stream) {
        ++input_reuse.hits;
    } else {
        hipLaunchKernelGGL(prepare_input_kernel, dim3(tiles_h, static_cast<unsigned>(m / 32)), dim3(32), 0, stream, xp, x_sums, x_tiles, static_cast<int>(h), tiles_h);
        check_launch("gemm_w4a16_ffn input");
    }
    hipLaunchKernelGGL(gemm_w4a16_kernel<true>, dim3(static_cast<unsigned>(n / kTileN), static_cast<unsigned>(m / kBlockDim)), dim3(256), 0, stream, act_tiles, x_tiles, x_sums,
                       gate_up_weights, static_cast<int>(2 * n), tiles_h, tiles_h, static_cast<int>(2 * stored / kTileN), act_half_sums);
    check_launch("gemm_w4a16_ffn gate and up");
    hipLaunchKernelGGL(add_half_sums_kernel, dim3(static_cast<unsigned>(m * tiles_act / 128)), dim3(128), 0, stream, act_half_sums, act_sums);
    check_launch("gemm_w4a16_ffn sums");
    hipLaunchKernelGGL(gemm_w4a16_kernel<false>, dim3(static_cast<unsigned>(h / kBlockDim), static_cast<unsigned>(m / kBlockDim)), dim3(256), 0, stream, yp, act_tiles, act_sums,
                       down_weights, static_cast<int>(h), tiles_act, static_cast<int>(stored / kTileK), static_cast<int>(h / kTileN), static_cast<float *>(nullptr));
    check_launch("gemm_w4a16_ffn down");
}

} // namespace heteromega

// Latency of the GPU slice of one FFN sublayer for a grid of (M, s).
// Weights are random bytes in the packed layout with valid scales, so timing matches real weights.
//
// Usage: bench_gpu_ffn_slice [--m 1,128,512,2048,4096] [--s 1024,2048,...,14336] [--iters 50] [--warmup 10] [--prefix]
//   --prefix  pack all 14336 channels once (interleaved gate/up) and run each s as FfnSliceWeights::prefix(s),
//             the way one weight copy serves a different split point in prefill and in decode.
//             Without it, each s gets its own packed slice (blocked gate/up).
// Output: CSV on stdout. s = 14336 is the GPU-only run of the same kernels (ablation A1).
#include "bench_common.hpp"
#include "bench_weights.hpp"
#include "heteromega/gpu_ffn.hpp"

#include <c10/hip/HIPStream.h>

#include <cstdio>

using namespace heteromega;

namespace {

constexpr int64_t kHidden = 4096;
constexpr int64_t kInter = 14336;

std::string default_splits() {
    std::string s;
    for (int64_t v = 1024; v <= 13312; v += 1024) {
        s += std::to_string(v) + ",";
    }
    return s + std::to_string(kInter);
}

} // namespace

int main(int argc, char **argv) {
    const auto ms = bench::parse_list(bench::arg(argc, argv, "--m", "1,128,512,2048,4096"));
    const auto splits = bench::parse_list(bench::arg(argc, argv, "--s", default_splits()));
    const int iters = std::stoi(bench::arg(argc, argv, "--iters", "50"));
    const int warmup = std::stoi(bench::arg(argc, argv, "--warmup", "10"));
    const bool prefix = bench::flag(argc, argv, "--prefix");

    const torch::Device dev(torch::kCUDA, 0);
    const hipStream_t stream = c10::hip::getCurrentHIPStream().stream();
    const int64_t max_m = *std::max_element(ms.begin(), ms.end());

    FfnSliceWeights all;
    if (prefix) {
        all.gate_up = bench::random_packed(kHidden, 2 * kInter, dev);
        all.down = bench::random_packed(kInter, kHidden, dev);
        all.begin = 0, all.end = kInter, all.hidden = kHidden, all.layout = GateUpLayout::kInterleaved;
    }
    auto all_buf = GpuFfnBuffers::allocate(max_m, prefix ? kInter : 1, kHidden, dev);

    std::printf("M,s,gpu_fraction,mode,gate_up_ms,silu_mul_ms,down_ms,total_ms,weight_GBps\n");
    for (int64_t s : splits) {
        if (s <= 0 || s > kInter || s % 512 != 0) {
            std::fprintf(stderr, "skip s=%lld: must be a multiple of 512 in (0, %lld]\n", static_cast<long long>(s),
                         static_cast<long long>(kInter));
            continue;
        }
        FfnSliceWeights w;
        if (prefix) {
            w = all.prefix(s);
        } else {
            w.gate_up = bench::random_packed(kHidden, 2 * s, dev);
            w.down = bench::random_packed(s, kHidden, dev);
            w.begin = 0, w.end = s, w.hidden = kHidden;
        }
        auto buf = prefix ? all_buf : GpuFfnBuffers::allocate(max_m, s, kHidden, dev);
        LinearOptions gate_up_opt, down_opt;
        gate_up_opt.packed_n = 2 * w.stored_channels();
        down_opt.packed_k = w.stored_channels();
        // Bytes the kernels read: the [4096, 2s] and [s, 4096] blocks, also when they sit in a larger packed matrix.
        const double weight_bytes = static_cast<double>(3 * s * kHidden / (kTileK * kTileN) * kTileBytes);

        for (int64_t m : ms) {
            // The GEMM reads whole 128-row tiles, so x gets padded storage (see padded_rows()).
            const auto x = (torch::randn({padded_rows(m), kHidden}, torch::TensorOptions().device(dev)) * 0.5)
                               .to(torch::kBFloat16)
                               .narrow(0, 0, m);
            auto y = torch::empty({m, kHidden}, x.options());
            auto gate_up = buf.gate_up.view(-1).narrow(0, 0, m * 2 * s).view({m, 2 * s});
            auto act = buf.act.view(-1).narrow(0, 0, m * s).view({m, s});
            const auto run = [&](bench::EventTimer &t, int it) {
                t.mark(4 * it);
                packed_linear(gate_up, x, w.gate_up, kHidden, 2 * s, stream, gate_up_opt);
                t.mark(4 * it + 1);
                silu_mul(gate_up, act, stream, w.layout);
                t.mark(4 * it + 2);
                packed_linear(y, act, w.down, s, kHidden, stream, down_opt);
                t.mark(4 * it + 3);
            };

            bench::EventTimer warm(stream, 4 * warmup);
            for (int it = 0; it < warmup; ++it) {
                run(warm, it);
            }
            bench::hip_check(hipStreamSynchronize(stream), "hipStreamSynchronize");
            // Enqueue all iterations back to back, then read the events.
            bench::EventTimer t(stream, 4 * iters);
            for (int it = 0; it < iters; ++it) {
                run(t, it);
            }
            std::vector<double> t_gu, t_act, t_down, t_total;
            for (int it = 0; it < iters; ++it) {
                const int b = 4 * it;
                t_gu.push_back(t.ms(b, b + 1));
                t_act.push_back(t.ms(b + 1, b + 2));
                t_down.push_back(t.ms(b + 2, b + 3));
                t_total.push_back(t.ms(b, b + 3));
            }
            const double gu = bench::median(t_gu), a = bench::median(t_act), d = bench::median(t_down), tot = bench::median(t_total);
            std::printf("%lld,%lld,%.4f,%s,%.4f,%.4f,%.4f,%.4f,%.1f\n", static_cast<long long>(m), static_cast<long long>(s),
                        static_cast<double>(s) / kInter, prefix ? "prefix" : "slice", gu, a, d, tot, weight_bytes / ((gu + d) * 1e6));
            std::fflush(stdout);
        }
    }
    return 0;
}

// Ablation of the GPU FFN changes against HeteroMosaic's FFN sequence, on the same weights and inputs.
//   hm      HeteroMosaic order (unified_llm_w4a16.cpp): gate GEMM, up GEMM, torch::silu_, mul_, down GEMM
//   merged  + gate and up as one GEMM. SiLU and multiply are still two Torch ops
//   fused   + one SiLU-and-multiply kernel. For M = 1 all three use HeteroMosaic's GEMV
//   gemv    fused with gemv_w4a16 instead of HeteroMosaic's GEMV: the gpu_ffn_slice_forward path. M = 1 only
//   dense   reference only: the same math with unquantized bf16 weights and torch::mm (hipBLASLt)
// No variant syncs the host between stages (HeteroMosaic waits on futures there), so the numbers compare kernels only.
//
// Each round runs every (s, M) once, with the variants in a new random order, so that clock and temperature drift
// do not favor the variant that runs first. Report the spread over rounds (tools/summarize_bench.py).
//
// Usage: bench_ffn_ablation [--m 1,128,512,2048,4096] [--s 10240,14336] [--variants hm,merged,fused,gemv,dense]
//                           [--rounds 5] [--iters 30] [--warmup 5] [--seed 1]
// Output: CSV on stdout.
//   gemm_TFLOPS  2*M*K*N of both matmuls over gemm1_ms + down_ms
//   weight_GBps  weight bytes read by both matmuls over gemm1_ms + down_ms (the decode metric)
//   act_GBps     bytes the activation step reads and writes: 5 passes over [M, s] for the Torch ops, 3 for the kernel
//   sclk_*, power_W, temp_C  GPU clock, SoC package power and GPU temperature while the timed iterations ran
#include "bench_common.hpp"
#include "bench_weights.hpp"
#include "heteromega/gpu_ffn.hpp"

#include <c10/hip/HIPStream.h>

#include <cstdio>
#include <functional>
#include <random>

using namespace heteromega;

namespace {

constexpr int64_t kHidden = 4096;
constexpr int64_t kInter = 14336;

struct Variant {
    std::string name;
    std::function<void()> gemm1, act, down;
    double act_passes;
    double weight_bytes;
};

struct SliceWeights {
    int64_t s;
    torch::Tensor gate, up, gate_up, down, dense_gate_up, dense_down;
};

// Contiguous [rows, width] view of the first rows * width elements of a flat buffer.
torch::Tensor leading(const torch::Tensor &flat, int64_t rows, int64_t width) { return flat.narrow(0, 0, rows * width).view({rows, width}); }

} // namespace

int main(int argc, char **argv) {
    const auto ms = bench::parse_list(bench::arg(argc, argv, "--m", "1,128,512,2048,4096"));
    const auto splits = bench::parse_list(bench::arg(argc, argv, "--s", "10240,14336"));
    const std::string wanted = "," + bench::arg(argc, argv, "--variants", "hm,merged,fused,gemv,dense") + ",";
    const int rounds = std::stoi(bench::arg(argc, argv, "--rounds", "5"));
    const int iters = std::stoi(bench::arg(argc, argv, "--iters", "30"));
    const int warmup = std::stoi(bench::arg(argc, argv, "--warmup", "5"));
    std::mt19937 rng(std::stoul(bench::arg(argc, argv, "--seed", "1")));

    const torch::Device dev(torch::kCUDA, 0);
    const hipStream_t stream = c10::hip::getCurrentHIPStream().stream();
    const int64_t max_rows = padded_rows(*std::max_element(ms.begin(), ms.end()));
    const auto bf16 = torch::TensorOptions().dtype(torch::kBFloat16).device(dev);
    const auto randn_bf16 = [&](int64_t rows, int64_t cols, double scale) {
        return (torch::randn({rows, cols}, torch::TensorOptions().device(dev)) * scale).to(torch::kBFloat16);
    };

    std::vector<SliceWeights> weights;
    int64_t max_s = 0;
    for (int64_t s : splits) {
        if (s <= 0 || s > kInter || s % 512 != 0) {
            std::fprintf(stderr, "skip s=%lld: must be a multiple of 512 in (0, %lld]\n", static_cast<long long>(s),
                         static_cast<long long>(kInter));
            continue;
        }
        weights.push_back({s, bench::random_packed(kHidden, s, dev), bench::random_packed(kHidden, s, dev),
                           bench::random_packed(kHidden, 2 * s, dev), bench::random_packed(s, kHidden, dev),
                           randn_bf16(kHidden, 2 * s, 0.02), randn_bf16(s, kHidden, 0.02)});
        max_s = std::max(max_s, s);
    }

    // Flat scratch with room for padded rows, as in GpuFfnBuffers: the GEMM reads whole 128-row tiles.
    auto gate_flat = torch::empty({max_rows * max_s}, bf16);
    auto up_flat = torch::empty({max_rows * max_s}, bf16);
    auto gate_up_flat = torch::empty({max_rows * 2 * max_s}, bf16);
    auto act_flat = torch::empty({max_rows * max_s}, bf16);

    LinearOptions hm_gemv;
    hm_gemv.gemv = GemvKernel::kHeteroMosaic;
    bench::SensorSampler sensors;
    if (!sensors.available()) {
        std::fprintf(stderr, "no GPU hwmon sensors found: sclk, power and temp columns are -1\n");
    }

    std::printf("round,pos,variant,M,s,gemm1_ms,act_ms,down_ms,total_ms,gemm_TFLOPS,weight_GBps,act_GBps,"
                "sclk_MHz_median,sclk_MHz_min,power_W,temp_C\n");
    for (int round = 0; round < rounds; ++round) {
        for (const auto &w : weights) {
            const int64_t s = w.s;
            for (int64_t m : ms) {
                const auto x = randn_bf16(padded_rows(m), kHidden, 0.5).narrow(0, 0, m);
                auto y = torch::empty({m, kHidden}, bf16);
                auto gate = leading(gate_flat, m, s);
                auto up = leading(up_flat, m, s);
                auto gate_up = leading(gate_up_flat, m, 2 * s);
                auto act = leading(act_flat, m, s);
                const auto gate_view = gate_up.narrow(1, 0, s);
                const auto up_view = gate_up.narrow(1, s, s);

                const double packed_split = static_cast<double>(w.gate.numel() + w.up.numel() + w.down.numel());
                const double packed_merged = static_cast<double>(w.gate_up.numel() + w.down.numel());
                const double dense = static_cast<double>(w.dense_gate_up.nbytes() + w.dense_down.nbytes());
                std::vector<Variant> variants = {
                    {"hm",
                     [&] {
                         packed_linear(gate, x, w.gate, kHidden, s, stream, hm_gemv);
                         packed_linear(up, x, w.up, kHidden, s, stream, hm_gemv);
                     },
                     [&] {
                         torch::silu_(gate);
                         gate.mul_(up);
                     },
                     [&] { packed_linear(y, gate, w.down, s, kHidden, stream, hm_gemv); }, 5.0, packed_split},
                    {"merged", [&] { packed_linear(gate_up, x, w.gate_up, kHidden, 2 * s, stream, hm_gemv); },
                     [&] {
                         at::silu_out(act, gate_view);
                         act.mul_(up_view);
                     },
                     [&] { packed_linear(y, act, w.down, s, kHidden, stream, hm_gemv); }, 5.0, packed_merged},
                    {"fused", [&] { packed_linear(gate_up, x, w.gate_up, kHidden, 2 * s, stream, hm_gemv); },
                     [&] { silu_mul(gate_up, act, stream); }, [&] { packed_linear(y, act, w.down, s, kHidden, stream, hm_gemv); }, 3.0,
                     packed_merged},
                    {"gemv", [&] { packed_linear(gate_up, x, w.gate_up, kHidden, 2 * s, stream); }, [&] { silu_mul(gate_up, act, stream); },
                     [&] { packed_linear(y, act, w.down, s, kHidden, stream); }, 3.0, packed_merged},
                    {"dense", [&] { at::mm_out(gate_up, x, w.dense_gate_up); }, [&] { silu_mul(gate_up, act, stream); },
                     [&] { at::mm_out(y, act, w.dense_down); }, 3.0, dense},
                };
                std::vector<Variant> order;
                for (auto &v : variants) {
                    if (wanted.find("," + v.name + ",") != std::string::npos && (v.name != "gemv" || m == 1)) {
                        order.push_back(v);
                    }
                }
                std::shuffle(order.begin(), order.end(), rng);

                for (size_t pos = 0; pos < order.size(); ++pos) {
                    const Variant &v = order[pos];
                    const auto run = [&](bench::EventTimer &t, int it) {
                        t.mark(4 * it);
                        v.gemm1();
                        t.mark(4 * it + 1);
                        v.act();
                        t.mark(4 * it + 2);
                        v.down();
                        t.mark(4 * it + 3);
                    };
                    bench::EventTimer warm(stream, 4 * warmup);
                    for (int it = 0; it < warmup; ++it) {
                        run(warm, it);
                    }
                    bench::hip_check(hipStreamSynchronize(stream), "hipStreamSynchronize");
                    // Enqueue all iterations back to back, then read the events.
                    sensors.start();
                    bench::EventTimer t(stream, 4 * iters);
                    for (int it = 0; it < iters; ++it) {
                        run(t, it);
                    }
                    std::vector<double> t_g1, t_act, t_down, t_total;
                    for (int it = 0; it < iters; ++it) {
                        const int b = 4 * it;
                        t_g1.push_back(t.ms(b, b + 1));
                        t_act.push_back(t.ms(b + 1, b + 2));
                        t_down.push_back(t.ms(b + 2, b + 3));
                        t_total.push_back(t.ms(b, b + 3));
                    }
                    const auto sens = sensors.stop();
                    const double g1 = bench::median(t_g1), a = bench::median(t_act), d = bench::median(t_down), tot = bench::median(t_total);
                    const double flops = 2.0 * m * kHidden * (2 * s) + 2.0 * m * s * kHidden;
                    const double act_bytes = v.act_passes * m * s * 2.0;
                    std::printf("%d,%zu,%s,%lld,%lld,%.4f,%.4f,%.4f,%.4f,%.2f,%.1f,%.1f,%.0f,%.0f,%.1f,%.0f\n", round, pos, v.name.c_str(),
                                static_cast<long long>(m), static_cast<long long>(s), g1, a, d, tot, flops / ((g1 + d) * 1e9),
                                v.weight_bytes / ((g1 + d) * 1e6), act_bytes / (a * 1e6), sens.sclk_mhz_median, sens.sclk_mhz_min,
                                sens.power_w_median, sens.temp_c_max);
                    std::fflush(stdout);
                }
            }
        }
    }
    return 0;
}

// Decode (M = 1) matmuls of the FFN slice: gemv_w4a16 against HeteroMosaic's GEMV and a dense bf16 torch::mm.
//   hm       hipkernels::w4a16_gemv_fused_packed
//   ours_kN  gemv_w4a16 with split_k = N (k0 = the automatic choice)
//   prefix   gemv_w4a16 (automatic split) on the leading [k, n] block of the whole FFN's packed matrix
//            (gate_up: n of 2 * 14336 columns; down: k of 14336 rows), as FfnSliceWeights::prefix() uses it
//   dense    torch::mm with bf16 weights
//
// Usage: bench_gemv [--s 10240,14336] [--iters 200] [--warmup 20]
// Output: CSV on stdout. weight_GBps = bytes of the weights the kernel reads / time.
#include "bench_common.hpp"
#include "bench_weights.hpp"
#include "heteromega/gpu_ffn.hpp"

#include <c10/hip/HIPStream.h>

#include <cstdio>
#include <functional>

using namespace heteromega;

namespace {

constexpr int64_t kHidden = 4096;
constexpr int64_t kInter = 14336;

} // namespace

int main(int argc, char **argv) {
    const auto splits = bench::parse_list(bench::arg(argc, argv, "--s", "10240,14336"));
    const int iters = std::stoi(bench::arg(argc, argv, "--iters", "200"));
    const int warmup = std::stoi(bench::arg(argc, argv, "--warmup", "20"));

    const torch::Device dev(torch::kCUDA, 0);
    const hipStream_t stream = c10::hip::getCurrentHIPStream().stream();
    const auto bf16 = torch::TensorOptions().dtype(torch::kBFloat16).device(dev);
    const auto full_gate_up = bench::random_packed(kHidden, 2 * kInter, dev);
    const auto full_down = bench::random_packed(kInter, kHidden, dev);
    bench::SensorSampler sensors;

    std::printf("matrix,s,k,n,kernel,ms,weight_GBps,sclk_MHz_median\n");
    for (int64_t s : splits) {
        struct Shape {
            const char *name;
            int64_t k, n;
            const torch::Tensor *full; // whole-FFN packed matrix for the prefix kernel
            int64_t full_k, full_n;
        };
        const Shape shapes[] = {{"gate_up", kHidden, 2 * s, &full_gate_up, kHidden, 2 * kInter},
                                {"down", s, kHidden, &full_down, kInter, kHidden}};
        for (const auto &sh : shapes) {
            const auto packed = bench::random_packed(sh.k, sh.n, dev);
            const auto dense_w = (torch::randn({sh.k, sh.n}, torch::TensorOptions().device(dev)) * 0.02).to(torch::kBFloat16);
            const auto x = (torch::randn({1, sh.k}, torch::TensorOptions().device(dev)) * 0.5).to(torch::kBFloat16);
            auto y = torch::empty({1, sh.n}, bf16);
            const double packed_bytes = static_cast<double>(packed.numel());

            struct Kernel {
                std::string name;
                std::function<void()> run;
                double bytes;
            };
            LinearOptions hm;
            hm.gemv = GemvKernel::kHeteroMosaic;
            std::vector<Kernel> kernels = {{"hm", [&] { packed_linear(y, x, packed, sh.k, sh.n, stream, hm); }, packed_bytes}};
            for (int split : {0, 1, 2, 4, 8}) {
                kernels.push_back({"ours_k" + std::to_string(split),
                                   [&, split] { gemv_w4a16(y, x, packed, sh.k, sh.n, sh.k, sh.n, split, stream); }, packed_bytes});
            }
            kernels.push_back({"prefix", [&] { gemv_w4a16(y, x, *sh.full, sh.k, sh.n, sh.full_k, sh.full_n, 0, stream); }, packed_bytes});
            kernels.push_back({"dense", [&] { at::mm_out(y, x, dense_w); }, static_cast<double>(dense_w.nbytes())});

            for (const auto &kn : kernels) {
                for (int it = 0; it < warmup; ++it) {
                    kn.run();
                }
                bench::hip_check(hipStreamSynchronize(stream), "hipStreamSynchronize");
                sensors.start();
                bench::EventTimer t(stream, iters + 1);
                t.mark(0);
                for (int it = 0; it < iters; ++it) {
                    kn.run();
                    t.mark(it + 1);
                }
                std::vector<double> times;
                for (int it = 0; it < iters; ++it) {
                    times.push_back(t.ms(it, it + 1));
                }
                const auto sens = sensors.stop();
                const double ms = bench::median(times);
                std::printf("%s,%lld,%lld,%lld,%s,%.4f,%.1f,%.0f\n", sh.name, static_cast<long long>(s), static_cast<long long>(sh.k),
                            static_cast<long long>(sh.n), kn.name.c_str(), ms, kn.bytes / (ms * 1e6), sens.sclk_mhz_median);
                std::fflush(stdout);
            }
        }
    }
    return 0;
}

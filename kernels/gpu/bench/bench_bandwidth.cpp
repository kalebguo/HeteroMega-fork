// GPU streaming-read bandwidth, alone or next to other readers of the same DRAM. No LibTorch.
//
// 1. Run alone to get BW_GPU_alone.
// 2. Run with --cpu-threads N: CPU threads read their own buffer at the same time. The output then has both shares.
// 3. Start an NPU load (for example an NPU read benchmark), then run this again to get the GPU share of BW_both.
//    Run with --no-gpu --cpu-threads N to get BW_CPU_alone.
//
// Usage: bench_bandwidth [--mib 1024] [--seconds 10] [--host] [--cpu-threads 0] [--cpu-mib 1024] [--no-gpu]
//   --host         read pinned host memory (hipHostMalloc) instead of device memory (hipMalloc)
//   --cpu-threads  CPU threads that stream-read a separate --cpu-mib buffer while the GPU reads
//   --no-gpu       CPU readers only
// Output: one line per ~1 s window ("t_s,GBps,cpu_GBps,sclk_MHz,power_W"), then the medians.
//   GBps is the GPU's read rate. power_W is the SoC package power (PPT), which the GPU, NPU and CPU share.
#include "bench_common.hpp"

#include <chrono>
#include <cstdio>
#include <memory>

namespace {

__global__ void __launch_bounds__(256) read_kernel(const uint4 *__restrict__ p, size_t n, unsigned *sink) {
    unsigned acc = 0;
    for (size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x; i < n; i += static_cast<size_t>(gridDim.x) * blockDim.x) {
        const uint4 v = p[i];
        acc ^= v.x ^ v.y ^ v.z ^ v.w;
    }
    if (acc == 0x9e3779b9u) {
        sink[0] = acc; // keeps the loads alive
    }
}

// CPU threads that read their slices of one buffer in a loop and count the bytes.
class CpuReaders {
  public:
    CpuReaders(int threads, size_t bytes) : words_(bytes / sizeof(uint64_t)), data_(new uint64_t[words_]), counts_(threads) {
        for (size_t i = 0; i < words_; ++i) {
            data_[i] = i * 0x9e3779b97f4a7c15ull; // touch every page before timing
        }
        for (int t = 0; t < threads; ++t) {
            threads_.emplace_back([this, t, threads] {
                const size_t begin = words_ * t / threads, end = words_ * (t + 1) / threads;
                uint64_t acc = 0;
                while (running_) {
                    for (size_t i = begin; i < end; ++i) {
                        acc ^= data_[i];
                    }
                    counts_[t] += (end - begin) * sizeof(uint64_t);
                }
                if (acc == 0x9e3779b9u) {
                    std::fprintf(stderr, "%llu\n", static_cast<unsigned long long>(acc)); // keeps the loads alive
                }
            });
        }
    }
    ~CpuReaders() {
        running_ = false;
        for (auto &t : threads_) {
            t.join();
        }
    }
    uint64_t bytes() const {
        uint64_t b = 0;
        for (const auto &c : counts_) {
            b += c.load();
        }
        return b;
    }

  private:
    size_t words_;
    std::unique_ptr<uint64_t[]> data_;
    std::vector<std::atomic<uint64_t>> counts_;
    std::vector<std::thread> threads_;
    std::atomic<bool> running_{true};
};

} // namespace

int main(int argc, char **argv) {
    const size_t mib = std::stoull(bench::arg(argc, argv, "--mib", "1024"));
    const double seconds = std::stod(bench::arg(argc, argv, "--seconds", "10"));
    const bool host = bench::flag(argc, argv, "--host");
    const bool gpu = !bench::flag(argc, argv, "--no-gpu");
    const int cpu_threads = std::stoi(bench::arg(argc, argv, "--cpu-threads", "0"));
    const size_t cpu_bytes = std::stoull(bench::arg(argc, argv, "--cpu-mib", "1024")) << 20;
    const size_t bytes = mib << 20;
    const size_t n = bytes / sizeof(uint4);

    void *buf = nullptr;
    unsigned *sink = nullptr;
    hipStream_t stream = nullptr;
    dim3 grid(1), block(256);
    if (gpu) {
        if (host) {
            bench::hip_check(hipHostMalloc(&buf, bytes, hipHostMallocDefault), "hipHostMalloc");
        } else {
            bench::hip_check(hipMalloc(&buf, bytes), "hipMalloc");
        }
        bench::hip_check(hipMemset(buf, 1, bytes), "hipMemset");
        bench::hip_check(hipMalloc(&sink, sizeof(unsigned)), "hipMalloc");

        hipDeviceProp_t prop;
        bench::hip_check(hipGetDeviceProperties(&prop, 0), "hipGetDeviceProperties");
        grid = dim3(prop.multiProcessorCount * 16);
        std::fprintf(stderr, "%s, %zu MiB, %s memory, grid %u x 256\n", prop.name, mib, host ? "pinned host" : "device", grid.x);

        bench::hip_check(hipStreamCreate(&stream), "hipStreamCreate");
        for (int i = 0; i < 3; ++i) {
            hipLaunchKernelGGL(read_kernel, grid, block, 0, stream, static_cast<const uint4 *>(buf), n, sink);
        }
        bench::hip_check(hipStreamSynchronize(stream), "warmup");
    }
    std::unique_ptr<CpuReaders> cpu;
    if (cpu_threads > 0) {
        std::fprintf(stderr, "%d CPU reader threads, %zu MiB\n", cpu_threads, cpu_bytes >> 20);
        cpu = std::make_unique<CpuReaders>(cpu_threads, cpu_bytes);
    }
    bench::SensorSampler sensors;

    std::printf("t_s,GBps,cpu_GBps,sclk_MHz,power_W\n");
    std::vector<double> windows, cpu_windows, sclk_windows, power_windows;
    const auto start = std::chrono::steady_clock::now();
    auto win_start = start;
    size_t win_bytes = 0;
    uint64_t cpu_mark = cpu ? cpu->bytes() : 0;
    sensors.start();
    while (true) {
        if (gpu) {
            hipLaunchKernelGGL(read_kernel, grid, block, 0, stream, static_cast<const uint4 *>(buf), n, sink);
            bench::hip_check(hipStreamSynchronize(stream), "read_kernel");
            win_bytes += bytes;
        } else {
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
        const auto now = std::chrono::steady_clock::now();
        const double win_s = std::chrono::duration<double>(now - win_start).count();
        if (win_s >= 1.0) {
            const uint64_t cpu_now = cpu ? cpu->bytes() : 0;
            const double gbps = win_bytes / win_s / 1e9, cpu_gbps = (cpu_now - cpu_mark) / win_s / 1e9;
            const auto sens = sensors.stop();
            windows.push_back(gbps), cpu_windows.push_back(cpu_gbps);
            sclk_windows.push_back(sens.sclk_mhz_median), power_windows.push_back(sens.power_w_median);
            std::printf("%.1f,%.1f,%.1f,%.0f,%.1f\n", std::chrono::duration<double>(now - start).count(), gbps, cpu_gbps,
                        sens.sclk_mhz_median, sens.power_w_median);
            std::fflush(stdout);
            win_start = now;
            win_bytes = 0;
            cpu_mark = cpu_now;
            sensors.start();
        }
        if (std::chrono::duration<double>(now - start).count() >= seconds) {
            break;
        }
    }
    sensors.stop();
    std::printf("median,%.1f,%.1f,%.0f,%.1f\n", bench::median(windows), bench::median(cpu_windows), bench::median(sclk_windows),
                bench::median(power_windows));

    cpu.reset();
    if (gpu) {
        (void)hipStreamDestroy(stream);
        (void)hipFree(sink);
        if (host) {
            (void)hipHostFree(buf);
        } else {
            (void)hipFree(buf);
        }
    }
    return 0;
}

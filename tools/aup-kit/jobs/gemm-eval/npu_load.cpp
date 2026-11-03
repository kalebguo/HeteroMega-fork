// Runs the merged FFN NPU artifact in a loop with no GPU work, to measure what the NPU does to the GPU. Diagnostic only, runs until killed.
// npu_load DEPLOYMENT.json [idle_ms between runs]. NPU_LOAD_THP=1 makes its buffers from 2 MB pages through user pointers.
#include "heteromega/merged_ffn_contract.hpp"
#include <xrt/xrt_bo.h>
#include <xrt/xrt_device.h>
#include <xrt/xrt_hw_context.h>
#include <xrt/xrt_kernel.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <cstring>
#include <random>
#include <string>
#include <sys/mman.h>
#include <thread>
#include <vector>

static const bool use_thp = std::getenv("NPU_LOAD_THP") != nullptr;

static xrt::bo make_bo(xrt::device &device, size_t bytes, xrt::memory_group group) {
    if (!use_thp) return xrt::bo(device, bytes, xrt::bo::flags::host_only, group);
    const size_t huge = size_t(2) << 20, size = (bytes + huge - 1) / huge * huge;
    void *p = nullptr;
    if (posix_memalign(&p, huge, size) != 0) std::abort();
    if (madvise(p, size, MADV_HUGEPAGE) != 0) std::perror("madvise");
    std::memset(p, 0, size);
    return xrt::bo(device, p, size, xrt::bo::flags::host_only, group);
}

static void print_huge() {
    std::ifstream f("/proc/self/smaps_rollup");
    std::string line;
    while (std::getline(f, line)) {
        if (line.rfind("AnonHugePages", 0) == 0 || line.rfind("Rss:", 0) == 0) std::printf("npu_load %s\n", line.c_str());
    }
}

int main(int argc, char **argv) {
    const auto c = heteromega::MergedFfnConfig::load(argv[1]);
    const int idle_ms = argc > 2 ? std::atoi(argv[2]) : 0;
    xrt::device device(c.device_index);
    xrt::xclbin binary((c.artifact_dir / "final.xclbin").string());
    device.register_xclbin(binary);
    xrt::hw_context context(device, binary.get_uuid());
    xrt::kernel kernel(context, "MLIR_AIE");
    std::ifstream file(c.artifact_dir / "insts.bin", std::ios::binary | std::ios::ate);
    std::vector<char> bytes(static_cast<size_t>(file.tellg()));
    file.seekg(0);
    file.read(bytes.data(), bytes.size());
    xrt::bo instructions(device, bytes.size(), xrt::bo::flags::cacheable, kernel.group_id(1));
    instructions.write(bytes.data(), bytes.size(), 0);
    instructions.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    xrt::bo x = make_bo(device, c.activation_bytes(), kernel.group_id(3));
    xrt::bo z = make_bo(device, c.intermediate_bytes(), kernel.group_id(7));
    xrt::bo y = make_bo(device, c.activation_bytes(), kernel.group_id(8));
    std::mt19937_64 rng(1);
    auto *xp = x.map<uint16_t *>();
    for (size_t i = 0; i < c.activation_bytes() / 2; ++i) xp[i] = 0x3f00 + (rng() & 0xff); // Small BF16 values.
    x.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    std::vector<xrt::bo> weights;
    for (int i = 0; i < 3; ++i) {
        weights.push_back(make_bo(device, c.weight_bytes(), kernel.group_id(4 + i)));
        auto *w = weights.back().map<uint64_t *>();
        for (size_t j = 0; j < c.weight_bytes() / 8; ++j) w[j] = rng();
        weights.back().sync(XCL_BO_SYNC_BO_TO_DEVICE);
    }
    print_huge();
    std::printf("npu_load ready: %zu weight bytes per matrix\n", static_cast<size_t>(c.weight_bytes()));
    std::fflush(stdout);
    for (long n = 0;; ++n) {
        const auto start = std::chrono::steady_clock::now();
        auto run = kernel(3, instructions, static_cast<uint32_t>(bytes.size()), x, weights[0], weights[1], weights[2], z, y);
        run.wait(c.timeout_ms);
        if (n % 200 == 0) {
            const std::chrono::duration<double, std::milli> ms = std::chrono::steady_clock::now() - start;
            std::printf("npu_load run %ld: %.2f ms\n", n, ms.count());
            std::fflush(stdout);
        }
        if (idle_ms > 0) std::this_thread::sleep_for(std::chrono::milliseconds(idle_ms));
    }
}

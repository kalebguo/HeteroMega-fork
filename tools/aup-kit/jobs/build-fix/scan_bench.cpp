// CPU only: one scan of a finite [512,4096] BF16 buffer, at the flags of the heteromega_merged_ffn target (-O3, no -march).
#include "heteromega/merged_ffn_contract.hpp"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <vector>

// Same detection as the full scan with no per-element branch, so the compiler can vectorize it.
__attribute__((noinline)) bool any_nonfinite_branchless(const uint16_t *bits, int64_t count) {
    uint16_t seen = 0;
    for (int64_t i = 0; i < count; ++i) seen |= uint16_t((bits[i] & 0x7f80) == 0x7f80);
    return seen != 0;
}

template <class F> void report(const char *name, F scan) {
    std::vector<double> ms;
    for (int r = 0; r < 101; ++r) {
        const auto a = std::chrono::steady_clock::now();
        scan();
        ms.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - a).count());
    }
    std::sort(ms.begin(), ms.end());
    std::printf("%-22s median=%.4f ms min=%.4f max=%.4f\n", name, ms[ms.size() / 2], ms.front(), ms.back());
}

int main() {
    using heteromega::FiniteCheck;
    using heteromega::find_nonfinite_bf16;
    const int64_t rows = 512, columns = 4096;
    std::vector<uint16_t> bits(rows * columns, 0x3f80);
    volatile int64_t sink = 0;
    report("full", [&] { sink = find_nonfinite_bf16(bits.data(), rows, columns, FiniteCheck::kFull); });
    report("sampled", [&] { sink = find_nonfinite_bf16(bits.data(), rows, columns, FiniteCheck::kSampled); });
    report("full_branchless", [&] { sink = any_nonfinite_branchless(bits.data(), rows * columns); });
    return int(sink) == -1 || sink == 0 ? 0 : 1;
}

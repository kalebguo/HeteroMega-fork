#pragma once

// Deployment ABI checks are deliberately independent of Torch, HIP and XRT.
#include "third_party/nlohmann/json.hpp"
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>

namespace heteromega {
// HETEROMEGA_FFN_FINITE_CHECK. full (default): every element of the input and the NPU output on the critical path.
// sampled: every row at one in kFiniteSampleStride columns. A nonfinite element at an unsampled column of layer L is
// caught by layer L+1's input check, because the attention GEMMs spread it over its whole row; layer L=last relies on
// the logits/token checks instead.
enum class FiniteCheck { kFull, kSampled };
constexpr int64_t kFiniteSampleStride = 64;

inline FiniteCheck parse_finite_check(const char *value) {
    if (value == nullptr || *value == '\0' || std::string(value) == "full") return FiniteCheck::kFull;
    if (std::string(value) == "sampled") return FiniteCheck::kSampled;
    throw std::invalid_argument(std::string("merged FFN: HETEROMEGA_FFN_FINITE_CHECK must be full or sampled, not ") + value);
}

// Row-major BF16 bits. Returns the first checked nonfinite element (all 8 exponent bits set), or -1.
inline int64_t find_nonfinite_bf16(const uint16_t *bits, int64_t rows, int64_t columns, FiniteCheck mode) {
    const int64_t stride = mode == FiniteCheck::kFull ? 1 : kFiniteSampleStride;
    const int64_t count = rows * columns;
    if (stride == 1) {
        for (int64_t i = 0; i < count; ++i)
            if ((bits[i] & 0x7f80) == 0x7f80) return i;
        return -1;
    }
    for (int64_t row = 0; row < rows; ++row)
        for (int64_t i = row * columns; i < (row + 1) * columns; i += stride)
            if ((bits[i] & 0x7f80) == 0x7f80) return i;
    return -1;
}

struct MergedFfnConfig {
    std::filesystem::path artifact_dir;
    int64_t m = 0, hidden = 0, intermediate = 0, npu_channels = 0;
    unsigned timeout_ms = 10000;
    unsigned device_index = 0;
    std::string xclbin_sha256, insts_sha256;

    static size_t checked_product(size_t a, size_t b) {
        constexpr size_t limit = std::numeric_limits<size_t>::max() < uint64_t(INT64_MAX)
                                     ? std::numeric_limits<size_t>::max() : size_t(INT64_MAX);
        if (a == 0 || b == 0 || a > limit / b)
            throw std::invalid_argument("merged FFN: invalid or overflowing buffer size");
        return a * b;
    }
    size_t activation_bytes() const { return checked_product(checked_product(m, hidden), 2); }
    size_t intermediate_bytes() const { return checked_product(checked_product(m, npu_channels), 2); }
    size_t weight_bytes() const { return checked_product(checked_product(hidden / 128, intermediate / 64), 4352); }
    bool supports(int64_t rows, int64_t h, int64_t inter) const {
        return rows > 1 && rows == m && h == hidden && inter == intermediate;
    }

    static MergedFfnConfig load(const std::filesystem::path &path) {
        using Json = nlohmann::json;
        const auto read = [](const std::filesystem::path &p) {
            std::ifstream file(p);
            if (!file) throw std::runtime_error("merged FFN: cannot read " + p.string());
            Json j;
            file >> j;
            if (!j.is_object()) throw std::invalid_argument("merged FFN: expected JSON object");
            return j;
        };
        const auto positive = [](const Json &j, const char *key) -> int64_t {
            const auto &v = j.at(key);
            if (!v.is_number_integer() || (v.is_number_unsigned() && v.get<uint64_t>() > uint64_t(INT64_MAX)))
                throw std::invalid_argument(std::string("merged FFN: invalid integer ") + key);
            const auto result = v.get<int64_t>();
            if (result <= 0) throw std::invalid_argument(std::string("merged FFN: nonpositive ") + key);
            return result;
        };
        const auto config = read(path);
        // Explicit acceptance of the existing BFP16 deviations; this is not an accuracy claim.
        if (config.at("accuracy_policy") != "report")
            throw std::invalid_argument("merged FFN: this experimental kernel requires accuracy_policy=report");
        for (auto it = config.begin(); it != config.end(); ++it) {
            if (it.key() != "artifact_dir" && it.key() != "accuracy_policy" &&
                it.key() != "timeout_ms" && it.key() != "device_index")
                throw std::invalid_argument("merged FFN: unknown setting " + it.key());
        }
        MergedFfnConfig c;
        const auto root = config.at("artifact_dir").get<std::string>();
        if (root.empty()) throw std::invalid_argument("merged FFN: empty artifact_dir");
        c.artifact_dir = std::filesystem::absolute(path).parent_path() / root;
        if (config.contains("timeout_ms")) {
            const auto timeout = positive(config, "timeout_ms");
            if (timeout > 60000) throw std::invalid_argument("merged FFN: timeout_ms exceeds 60000");
            c.timeout_ms = static_cast<unsigned>(timeout);
        }
        if (config.contains("device_index")) {
            const auto &index = config.at("device_index");
            // Existing direct-XDNA runtime and first deployment target device zero.
            if (!index.is_number_integer() || index != 0)
                throw std::invalid_argument("merged FFN: only device_index=0 is supported");
        }
        const auto j = read(c.artifact_dir / "final.xclbin.json");
        const Json expected = {
            {"kind", "w4a16_upstream_merged"}, {"mode", "merged"},
            {"arithmetic", "bfp16"}, {"compute_format", "bfp16"}, {"accumulator_format", "bf16"},
            {"input_layout", "row_major"}, {"intermediate_layout", "row_major"}, {"output_layout", "row_major"},
            {"weight_tile_bytes", 4352}, {"cols", 8}, {"compute_rows", 4}, {"worker_count", 32},
            {"m_reuse", 4}, {"activation_m_reuse", 2}, {"segment_pipeline_depth", 2},
            {"activation_fifo_depth", 2}, {"z_output_fifo_depth", 1}, {"y_output_fifo_depth", 1},
            {"bfp16_native_layout", true}, {"split_m32_a", true}, {"preconvert_b", true},
            {"preconvert_a", false}, {"vector_zero", true}, {"mmul_m_reuse", 1},
            {"target_mlir_aie_commit", "7e00b57955e108fe9d8e9419f5828a0c7e650858"}};
        for (auto it = expected.begin(); it != expected.end(); ++it) {
            if (!j.contains(it.key()) || j.at(it.key()) != it.value())
                throw std::invalid_argument("merged FFN: unsupported artifact field " + it.key());
        }
        c.m = positive(j, "m");
        c.hidden = positive(j, "hidden");
        c.npu_channels = positive(j, "channels");
        c.intermediate = positive(j, "weight_n");
        if (c.m % 512 || c.hidden % 128 || c.intermediate % 128 || c.npu_channels % 256 ||
            c.npu_channels >= c.intermediate || c.npu_channels > 8192 ||
            positive(j, "weight_k") != c.hidden || positive(j, "weight_k_down") != c.intermediate ||
            positive(j, "weight_n_down") != c.hidden)
            throw std::invalid_argument("merged FFN: incompatible shape or channel partition");
        const auto digest = [&j](const char *key) {
            const auto hash = j.at(key).get<std::string>();
            if (hash.size() != 64 || hash.find_first_not_of("0123456789abcdef") != std::string::npos)
                throw std::invalid_argument(std::string("merged FFN: invalid SHA256 ") + key);
            return hash;
        };
        c.xclbin_sha256 = digest("xclbin_sha256");
        c.insts_sha256 = digest("insts_sha256");
        (void)c.activation_bytes();
        (void)c.intermediate_bytes();
        (void)c.weight_bytes();
        return c;
    }
};
} // namespace heteromega

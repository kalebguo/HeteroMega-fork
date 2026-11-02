#include "heteromega/merged_ffn_contract.hpp"
#include <functional>
#include <iostream>
#include <vector>

using Json = nlohmann::json;
using heteromega::MergedFfnConfig;
namespace fs = std::filesystem;

int main(int argc, char **argv) {
    if (argc == 2) {
        try {
            const auto c = MergedFfnConfig::load(argv[1]);
            std::cout << "PASS: deployment metadata M=" << c.m << " H=" << c.hidden
                      << " I=" << c.intermediate << " P=" << c.npu_channels << '\n';
            return 0;
        } catch (const std::exception &e) { std::cerr << e.what() << '\n'; return 1; }
    }
    // CTest supplies a disposable build directory. Never touches real kernel artifacts.
    const fs::path root = fs::current_path() / "contract-fixture";
    fs::create_directories(root / "artifact");
    const auto write = [](const fs::path &path, const Json &j) { std::ofstream(path) << j.dump(); };
    const Json manifest = {
        {"kind", "w4a16_upstream_merged"}, {"mode", "merged"},
        {"m", 512}, {"hidden", 4096}, {"channels", 6400},
        {"weight_k", 4096}, {"weight_n", 14336}, {"weight_k_down", 14336}, {"weight_n_down", 4096},
        {"arithmetic", "bfp16"}, {"compute_format", "bfp16"}, {"accumulator_format", "bf16"},
        {"input_layout", "row_major"}, {"intermediate_layout", "row_major"}, {"output_layout", "row_major"},
        {"weight_tile_bytes", 4352}, {"cols", 8}, {"compute_rows", 4}, {"worker_count", 32},
        {"m_reuse", 4}, {"activation_m_reuse", 2}, {"segment_pipeline_depth", 2},
        {"activation_fifo_depth", 2}, {"z_output_fifo_depth", 1}, {"y_output_fifo_depth", 1},
        {"bfp16_native_layout", true}, {"split_m32_a", true}, {"preconvert_b", true},
        {"preconvert_a", false}, {"vector_zero", true}, {"mmul_m_reuse", 1},
        {"target_mlir_aie_commit", "7e00b57955e108fe9d8e9419f5828a0c7e650858"},
        {"xclbin_sha256", std::string(64, 'a')}, {"insts_sha256", std::string(64, 'b')}};
    const Json settings = {{"artifact_dir", "artifact"}, {"accuracy_policy", "report"}};
    const auto config_path = root / "deploy.json", manifest_path = root / "artifact/final.xclbin.json";
    const auto load = [&] { return MergedFfnConfig::load(config_path); };
    const auto expect = [](bool okay, const char *label) {
        if (!okay) throw std::runtime_error(std::string("FAILED: ") + label);
    };
    unsigned rejected = 0;
    const auto rejects = [&](const Json &config, const Json &artifact) {
        write(config_path, config); write(manifest_path, artifact);
        bool failed = false;
        try { (void)load(); } catch (const std::exception &) { failed = true; }
        expect(failed, "invalid deployment accepted");
        ++rejected;
    };
    try {
        write(config_path, settings); write(manifest_path, manifest);
        auto c = load();
        expect(c.supports(512, 4096, 14336), "matching prefill");
        expect(!c.supports(1, 4096, 14336) && !c.supports(256, 4096, 14336) &&
               !c.supports(512, 4096, 11008), "decode and unsupported shapes fall back");
        expect(c.activation_bytes() == 4194304 && c.intermediate_bytes() == 6553600 &&
               c.weight_bytes() == 31195136, "BO sizing");
        expect(fs::equivalent(c.artifact_dir, root / "artifact"), "relative artifact path");
        for (const char *key : {"kind", "mode", "input_layout", "intermediate_layout", "output_layout",
                                "arithmetic", "target_mlir_aie_commit"}) {
            auto j = manifest; j[key] = "wrong"; rejects(settings, j);
        }
        for (const char *key : {"weight_k", "weight_k_down", "weight_n_down", "weight_tile_bytes",
                                "m_reuse", "activation_m_reuse", "worker_count"}) {
            auto j = manifest; j[key] = 1; rejects(settings, j);
        }
        for (const auto &value : {Json(0), Json(-128), Json(true), Json(512.5), Json("512"),
                                Json(UINT64_MAX)}) {
            auto j = manifest; j["m"] = value; rejects(settings, j);
        }
        for (int p : {0, 128, 6401, 8448, 14336}) {
            auto j = manifest; j["channels"] = p; rejects(settings, j);
        }
        for (const char *key : {"preconvert_b", "bfp16_native_layout", "split_m32_a"}) {
            auto j = manifest; j[key] = false; rejects(settings, j);
        }
        { auto j = manifest; j["xclbin_sha256"] = "bad"; rejects(settings, j); }
        { auto j = manifest; j.erase("insts_sha256"); rejects(settings, j); }
        { auto j = manifest; j["hidden"] = int64_t(INT64_MAX / 128 * 128);
          j["weight_k"] = j["hidden"]; j["weight_n_down"] = j["hidden"];
          rejects(settings, j); }
        for (const auto &value : {Json(0), Json(-1), Json(60001), Json(true)}) {
            auto j = settings; j["timeout_ms"] = value; rejects(j, manifest);
        }
        { auto j = settings; j["accuracy_policy"] = "strict"; rejects(j, manifest); }
        { auto j = settings; j["device_index"] = 1; rejects(j, manifest); }
        { auto j = settings; j["device_index"] = false; rejects(j, manifest); }
        { auto j = settings; j["npu_channel"] = 6400; rejects(j, manifest); }
        { auto j = settings; j["artifact_dir"] = ""; rejects(j, manifest); }
        bool overflow = false;
        try { (void)MergedFfnConfig::checked_product(SIZE_MAX, 2); } catch (const std::invalid_argument &) { overflow = true; }
        expect(overflow, "overflow guard");
        write(config_path, settings); write(manifest_path, manifest);
        std::cout << "PASS: valid shape/partition/BO sizes, fallback selection, " << rejected
                  << " invalid contracts rejected, overflow guard\n";

        using heteromega::FiniteCheck;
        using heteromega::find_nonfinite_bf16;
        using heteromega::parse_finite_check;
        expect(parse_finite_check(nullptr) == FiniteCheck::kFull && parse_finite_check("") == FiniteCheck::kFull &&
               parse_finite_check("full") == FiniteCheck::kFull && parse_finite_check("sampled") == FiniteCheck::kSampled,
               "finite check modes");
        bool bad_mode = false;
        try { parse_finite_check("off"); } catch (const std::invalid_argument &) { bad_mode = true; }
        expect(bad_mode, "unknown finite check mode rejected");
        const int64_t rows = 4, columns = 256;
        std::vector<uint16_t> bits(rows * columns, 0x3f80); // BF16 1.0
        const auto scan = [&](FiniteCheck mode) { return find_nonfinite_bf16(bits.data(), rows, columns, mode); };
        expect(scan(FiniteCheck::kFull) == -1 && scan(FiniteCheck::kSampled) == -1, "finite data passes");
        for (const uint16_t nonfinite : {uint16_t(0x7f80), uint16_t(0xff80), uint16_t(0x7fc0), uint16_t(0xffff)}) {
            bits[2 * columns + 3] = nonfinite; // Row 2, unsampled column 3.
            expect(scan(FiniteCheck::kFull) == 2 * columns + 3, "full check finds an isolated nonfinite element");
            expect(scan(FiniteCheck::kSampled) == -1, "sampled check skips unsampled columns");
            bits[2 * columns + 3] = 0x3f80;
            bits[3 * columns + 64] = nonfinite; // Row 3, sampled column 64.
            expect(scan(FiniteCheck::kSampled) == 3 * columns + 64, "sampled check reads every row at stride columns");
            bits[3 * columns + 64] = 0x3f80;
        }
        for (int64_t i = columns; i < 2 * columns; ++i) bits[i] = 0x7fc0; // The whole of row 1, as the next layer sees it.
        expect(scan(FiniteCheck::kFull) == columns && scan(FiniteCheck::kSampled) == columns,
               "both checks find a whole nonfinite row");
        std::cout << "PASS: finite check modes\n";
    } catch (const std::exception &e) { std::cerr << e.what() << '\n'; return 1; }
    return 0;
}

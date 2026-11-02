// One native HeteroMosaic packed W4A16 GEMM for the NPU error probes.
// Follows experiments/ffn_partition/native_heteromosaic_bindings.cpp: it links an already
// built model extension and calls the original M-split dispatcher and GPU kernel. No copied
// kernels, no replacement dispatcher, no CMake target. One initialization per process.
#include "unified_llm_w4a16_hetero/npuSetup.hpp"
#include "unified_llm_w4a16_hetero/hetero_compute.hpp"
#include "hipkernels/w4a16_gemm_packed.hpp"
#include <c10/hip/HIPStream.h>
#include <torch/extension.h>
#include <pybind11/stl.h>
#include <chrono>
#include <limits>

namespace py = pybind11;
namespace {
using Clock = std::chrono::steady_clock;
uint64_t completed = 0;
bool initialized = false;
int64_t shape_m = 0, shape_k = 0, shape_n = 0;

void sync(hipStream_t stream) {
    auto status = hipStreamSynchronize(stream);
    TORCH_CHECK(status == hipSuccess, "HIP synchronization failed: ", hipGetErrorString(status));
}

void require_tensor(const torch::Tensor &tensor, int64_t rows, int64_t cols) {
    TORCH_CHECK(tensor.is_cuda() && tensor.is_contiguous() && tensor.scalar_type() == torch::kBFloat16 &&
                    tensor.dim() == 2 && tensor.size(0) == rows && tensor.size(1) == cols,
                "expected contiguous ROCm BF16 tensor [", rows, ",", cols, "]");
}

py::dict initialize(const std::string &config, int64_t m, int64_t k, int64_t n) {
    TORCH_CHECK(!initialized, "single-initialization adapter; use a fresh process per case");
    TORCH_CHECK(config_map.empty() && ptr_to_handle_map.empty(), "native runtime globals already populated");
    read_npu_config(config);
    TORCH_CHECK(hw_target == "hetero", "configuration must select hetero");
    TORCH_CHECK(use_packed_weights && pad_packed_weights == 0, "requires packed weights without padding");
    TORCH_CHECK(initialize_xdna_driver("/dev/accel/accel0") == 0, "XDNA initialization failed");
    init_npu();
    load_npu_kernels(xdna_drv_fd, config);
    init_gemm_resources();
    NPUKey key = {static_cast<int>(m), static_cast<int>(k), static_cast<int>(n), get_layer_id("gate"), 0};
    auto found = config_map.find(key);
    TORCH_CHECK(found != config_map.end(), "native config missing for ", m, "x", k, "x", n);
    const auto &v = found->second;
    TORCH_CHECK(v.npuM == m && v.npuK == k && v.npuN == n && v.split_type == -1 && v.hw_idx >= 0 && v.inst_idx >= 0 &&
                    v.hw_idx < static_cast<int>(hwctxt_array.size()) &&
                    v.inst_idx < static_cast<int>(instctxt_array.size()),
                "native config is not an all-rows M-split on the NPU");
    TORCH_CHECK(hwctxt_array[v.hw_idx].pdi_handle != 0 && instctxt_array[v.inst_idx].dpu_0_handle != 0 &&
                    instctxt_array[v.inst_idx].num_dpu_0_insts > 0,
                "native artifact context is incomplete");
    shape_m = m; shape_k = k; shape_n = n;
    initialized = true;
    py::dict selected;
    selected["npu_m"] = v.npuM; selected["npu_k"] = v.npuK; selected["npu_n"] = v.npuN;
    selected["hardware_context"] = v.hw_idx; selected["instruction_context"] = v.inst_idx;
    return selected;
}

uint32_t import_tensor(torch::Tensor tensor) {
    TORCH_CHECK(initialized && tensor.is_cuda() && tensor.is_contiguous(), "initialize first; contiguous GPU tensor");
    TORCH_CHECK(tensor.scalar_type() == torch::kBFloat16 || tensor.scalar_type() == torch::kUInt8,
                "only BF16 activations/outputs and uint8 packed weights are imported");
    auto handle = import_dma_buf_to_xdna(tensor.data_ptr(), tensor.numel(), tensor.element_size());
    auto found = ptr_to_handle_map.find(tensor.data_ptr());
    TORCH_CHECK(handle != 0 && handle < std::numeric_limits<int32_t>::max() && found != ptr_to_handle_map.end() &&
                    found->second == handle,
                "DMA-BUF import failed; GPU fallback is forbidden in the probe");
    return handle;
}

// npu=true: original M-split dispatcher with all rows on the NPU; a missing NPU future is an error.
// npu=false: the original packed GPU GEMM on the same bytes (GPU control).
py::dict run(torch::Tensor input, torch::Tensor output, torch::Tensor weight, bool npu) {
    TORCH_CHECK(initialized, "initialize must precede run");
    require_tensor(input, shape_m, shape_k);
    require_tensor(output, shape_m, shape_n);
    TORCH_CHECK(weight.is_cuda() && weight.is_contiguous() && weight.scalar_type() == torch::kUInt8 && weight.dim() == 1 &&
                    weight.numel() == (shape_k / 128) * (shape_n / 64) * 4352,
                "packed weight contract mismatch");
    TORCH_CHECK(weight.device() == input.device() && output.device() == input.device(), "tensors must share a device");
    auto stream = c10::hip::getCurrentHIPStream().stream();
    sync(stream);
    const auto before = completed;
    const auto start = Clock::now();
    if (npu) {
        auto future = hetero_matmul_out_gemm_packed_M(output, input, weight, shape_k, shape_n, "gate", 0);
        TORCH_CHECK(future.valid(), "native dispatcher fell back to the GPU: no NPU future");
        const int rc = future.get();
        TORCH_CHECK(rc == 0, "native NPU completion failed: rc=", rc);
        ++completed;
    } else {
        hipkernels::w4a16_gemm_fused_packed(output, input, weight, shape_k, shape_n, stream);
    }
    sync(stream);
    py::dict result;
    result["wall_us"] = std::chrono::duration<double, std::micro>(Clock::now() - start).count();
    result["completed_npu_dispatches"] = completed - before;
    return result;
}
} // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("initialize", &initialize);
    m.def("import_tensor", &import_tensor);
    m.def("run", &run);
    m.def("completed_npu_dispatches", []() { return completed; });
}

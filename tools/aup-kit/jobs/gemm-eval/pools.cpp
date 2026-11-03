// Where each allocation path puts its memory on this APU: the change of amdgpu's VRAM and GTT counters for a 256 MB allocation. Diagnostic only.
#include <hip/hip_runtime.h>
#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>
#include <cstdio>
#include <fstream>
#include <glob.h>
#include <string>
#include <vector>

static long counter(const char *name) {
    glob_t g;
    long total = 0;
    if (glob((std::string("/sys/class/drm/card*/device/") + name).c_str(), 0, nullptr, &g) == 0) {
        for (size_t i = 0; i < g.gl_pathc; ++i) {
            std::ifstream f(g.gl_pathv[i]);
            long v = 0;
            if (f >> v) total += v;
        }
        globfree(&g);
    }
    return total;
}
struct Before {
    long vram = counter("mem_info_vram_used"), gtt = counter("mem_info_gtt_used");
    void report(const char *what) const {
        printf("  %-28s vram +%6.1f MB, gtt +%6.1f MB\n", what, (counter("mem_info_vram_used") - vram) / 1048576.0, (counter("mem_info_gtt_used") - gtt) / 1048576.0);
    }
};
static std::vector<hsa_agent_t> gpus, cpus;
static hsa_status_t on_agent(hsa_agent_t a, void *) {
    hsa_device_type_t t;
    hsa_agent_get_info(a, HSA_AGENT_INFO_DEVICE, &t);
    (t == HSA_DEVICE_TYPE_GPU ? gpus : cpus).push_back(a);
    return HSA_STATUS_SUCCESS;
}
static std::vector<hsa_amd_memory_pool_t> pools;
static hsa_status_t on_pool(hsa_amd_memory_pool_t p, void *) {
    pools.push_back(p);
    return HSA_STATUS_SUCCESS;
}
int main() {
    const size_t bytes = size_t(256) << 20;
    int count = 0;
    hipGetDeviceCount(&count);
    printf("hip devices %d\n", count);
    hipDeviceProp_t prop;
    hipGetDeviceProperties(&prop, 0);
    printf("integrated %d, multiProcessMemorySpace %s, totalGlobalMem %.1f GB\n", prop.integrated, prop.name, prop.totalGlobalMem / 1073741824.0);
    const struct { const char *name; unsigned flag; } flags[] = {{"hipExtMalloc default", hipDeviceMallocDefault}, {"hipExtMalloc finegrained", hipDeviceMallocFinegrained},
                                                                 {"hipExtMalloc uncached", hipDeviceMallocUncached}, {"hipExtMalloc contiguous", hipDeviceMallocContiguous}};
    {
        Before b;
        void *p = nullptr;
        const hipError_t e = hipMalloc(&p, bytes);
        printf("hipMalloc status %d\n", e);
        b.report("hipMalloc");
        hipFree(p);
    }
    for (const auto &f : flags) {
        Before b;
        void *p = nullptr;
        const hipError_t e = hipExtMallocWithFlags(&p, bytes, f.flag);
        printf("%s status %d\n", f.name, e);
        b.report(f.name);
        if (p) hipFree(p);
    }
    {
        Before b;
        void *p = nullptr;
        const hipError_t e = hipHostMalloc(&p, bytes, hipHostMallocNonCoherent);
        printf("hipHostMalloc noncoherent status %d\n", e);
        b.report("hipHostMalloc noncoherent");
        if (p) hipHostFree(p);
    }
    hsa_init();
    hsa_iterate_agents(on_agent, nullptr);
    printf("agents: %zu gpu, %zu cpu\n", gpus.size(), cpus.size());
    auto scan = [&](hsa_agent_t agent, const char *owner) {
        pools.clear();
        hsa_amd_agent_iterate_memory_pools(agent, on_pool, nullptr);
        for (size_t i = 0; i < pools.size(); ++i) {
            hsa_amd_segment_t seg;
            uint32_t gflags = 0;
            size_t size = 0;
            bool allowed = false;
            hsa_amd_memory_pool_get_info(pools[i], HSA_AMD_MEMORY_POOL_INFO_SEGMENT, &seg);
            hsa_amd_memory_pool_get_info(pools[i], HSA_AMD_MEMORY_POOL_INFO_GLOBAL_FLAGS, &gflags);
            hsa_amd_memory_pool_get_info(pools[i], HSA_AMD_MEMORY_POOL_INFO_SIZE, &size);
            hsa_amd_memory_pool_get_info(pools[i], HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_ALLOWED, &allowed);
            printf("%s pool %zu: segment %d, global flags 0x%x, size %.1f GB, runtime alloc %d\n", owner, i, seg, gflags, size / 1073741824.0, allowed);
            if (seg != HSA_AMD_SEGMENT_GLOBAL || !allowed) continue;
            Before b;
            void *p = nullptr;
            const hsa_status_t s = hsa_amd_memory_pool_allocate(pools[i], bytes, 0, &p);
            printf("  allocate status 0x%x\n", s);
            b.report("hsa pool");
            if (p) hsa_amd_memory_pool_free(p);
        }
    };
    if (!gpus.empty()) scan(gpus[0], "gpu");
    if (!cpus.empty()) scan(cpus[0], "cpu");
    return 0;
}

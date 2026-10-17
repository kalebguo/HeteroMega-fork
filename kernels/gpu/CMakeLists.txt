# GPU slice of the HeteroMega FFN split (see README.md in this directory).
# A separate CMake project: it does not change the HeteroMosaic build in the repository root.
# It compiles the HeteroMosaic packed W4A16 GEMM/GEMV from this repository's src/hipkernels.
#
# On the AUP cloud, with the environment of docs/cloud-baseline.md (venv active, ROCM_PATH exported):
#   cmake -S kernels/gpu -B build/gpu && cmake --build build/gpu -j4
#
# Defaults, each one can be overridden with -D:
#   ROCM_PATH                $ROCM_PATH, else the newest /opt/rocm-*
#   CMAKE_HIP_COMPILER       ROCm's clang++ (the AUP image also has a stale HIP 5.7 under /usr)
#   CMAKE_HIP_ARCHITECTURES  the GPU that rocminfo reports: native gfx1150 on the HX 370,
#                            gfx1100 if HSA_OVERRIDE_GFX_VERSION=11.0.0 is set
#   HETEROMOSAIC_ROOT        the repository root
#   HETEROMEGA_GEMM_AUTOTUNE OFF: build HeteroMosaic's GEMM with AUTOTUNE=1 when ON (it then times its kernel
#                            variants at the first call of each shape and keeps the fastest; it prints one line each)
cmake_minimum_required(VERSION 3.25)
project(heteromega_gpu LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

get_filename_component(_hm_default "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)
set(HETEROMOSAIC_ROOT "${_hm_default}" CACHE PATH "HeteroMosaic source tree (default: the repository root)")
if(NOT EXISTS "${HETEROMOSAIC_ROOT}/src/hipkernels/w4a16_gemm_packed.cu")
    message(FATAL_ERROR "HeteroMosaic sources not found (HETEROMOSAIC_ROOT = '${HETEROMOSAIC_ROOT}'). "
                        "Pass -DHETEROMOSAIC_ROOT=<path>.")
endif()
message(STATUS "HeteroMosaic sources: ${HETEROMOSAIC_ROOT}")

if(DEFINED ENV{ROCM_PATH} AND EXISTS "$ENV{ROCM_PATH}/include/hip/hip_runtime.h")
    set(_rocm_default "$ENV{ROCM_PATH}")
else()
    file(GLOB _rocm_dirs LIST_DIRECTORIES true "/opt/rocm-*")
    list(SORT _rocm_dirs COMPARE NATURAL)
    list(POP_BACK _rocm_dirs _rocm_default)
endif()
set(ROCM_PATH "${_rocm_default}" CACHE PATH "ROCm installation")
if(NOT EXISTS "${ROCM_PATH}/include/hip/hip_runtime.h")
    message(FATAL_ERROR "ROCm not found (ROCM_PATH = '${ROCM_PATH}'). Pass -DROCM_PATH=<path>.")
endif()
message(STATUS "ROCm: ${ROCM_PATH}")
set(ENV{ROCM_PATH} "${ROCM_PATH}")
list(APPEND CMAKE_PREFIX_PATH "${ROCM_PATH}")
list(APPEND CMAKE_BUILD_RPATH "${ROCM_PATH}/lib")
# The AUP image also has HIP 5.7's libamdhip64.so in /lib/x86_64-linux-gnu, and the HIP driver's -lamdhip64 searched
# there before ROCm's lib. bench_bandwidth then loaded both runtimes and crashed in comgr. CMake drops
# link_directories() for ROCm's lib (it is an implicit HIP link directory), so pass -L explicitly; user -L comes first.
add_link_options("-L${ROCM_PATH}/lib")

if(NOT CMAKE_HIP_COMPILER AND NOT DEFINED ENV{HIPCXX} AND EXISTS "${ROCM_PATH}/lib/llvm/bin/clang++")
    set(CMAKE_HIP_COMPILER "${ROCM_PATH}/lib/llvm/bin/clang++")
endif()
# The AUP image also has HIP 5.7 headers in /usr/include/hip. Without this, HIP files picked them up and
# clang 22 failed on them (__AMDGCN_WAVEFRONT_SIZE). Search ROCm's headers first, as docs/cloud-baseline.md does.
string(APPEND CMAKE_HIP_FLAGS " --rocm-path=${ROCM_PATH} --hip-path=${ROCM_PATH} -isystem ${ROCM_PATH}/include")
string(APPEND CMAKE_CXX_FLAGS " -isystem ${ROCM_PATH}/include")

if(NOT CMAKE_HIP_ARCHITECTURES)
    find_program(_rocminfo rocminfo HINTS "${ROCM_PATH}/bin" NO_CACHE)
    if(_rocminfo)
        execute_process(COMMAND "${_rocminfo}" OUTPUT_VARIABLE _rocminfo_out ERROR_QUIET)
        string(REGEX MATCH "gfx[0-9a-f]+" _gfx "${_rocminfo_out}")
    endif()
    if(NOT _gfx)
        message(FATAL_ERROR "rocminfo reports no GPU. Pass -DCMAKE_HIP_ARCHITECTURES=<gfx...>.")
    endif()
    set(CMAKE_HIP_ARCHITECTURES "${_gfx}" CACHE STRING "HIP architectures" FORCE)
endif()
message(STATUS "HIP architectures: ${CMAKE_HIP_ARCHITECTURES}")
enable_language(HIP)

# LibTorch from the active Python environment, as in HeteroMosaic.
find_package(Python COMPONENTS Interpreter REQUIRED)
execute_process(
    COMMAND ${Python_EXECUTABLE} -c "import torch, os; print(os.path.dirname(torch.__file__))"
    OUTPUT_VARIABLE TORCH_PYTHON_PATH
    OUTPUT_STRIP_TRAILING_WHITESPACE)
list(APPEND CMAKE_PREFIX_PATH "${TORCH_PYTHON_PATH}/share/cmake/Torch")
find_package(Torch REQUIRED)
find_package(hip REQUIRED)
find_package(OpenMP)
find_package(Threads REQUIRED)
option(HETEROMEGA_GEMM_AUTOTUNE "Build HeteroMosaic's GEMM with AUTOTUNE=1" OFF)

# The Torch ROCm wheel bundles its own HIP runtime and hipBLASLt in torch/lib, under names without a version
# (libamdhip64.so has SONAME libamdhip64.so.7), and libtorch_hip loads them by those names. Our executables need
# them by SONAME, which LD_LIBRARY_PATH resolved to ROCM_PATH. The process then had two copies of each, and
# librocroller crashed at exit ("double free or corruption"). So give Torch's copies their SONAMEs in the build
# tree and search there first. DT_RPATH (--disable-new-dtags) is searched before LD_LIBRARY_PATH; DT_RUNPATH is not.
set(TORCH_LIB_DIR "${TORCH_PYTHON_PATH}/lib")
set(TORCH_ROCM_ALIAS_DIR "${CMAKE_BINARY_DIR}/torch-rocm")
foreach(name libamdhip64.so libhipblaslt.so)
    if(CMAKE_READELF AND EXISTS "${TORCH_LIB_DIR}/${name}")
        execute_process(COMMAND "${CMAKE_READELF}" -d "${TORCH_LIB_DIR}/${name}" OUTPUT_VARIABLE _dyn ERROR_QUIET)
        if(_dyn MATCHES "soname: \\[([A-Za-z0-9_.+-]+)\\]")
            file(MAKE_DIRECTORY "${TORCH_ROCM_ALIAS_DIR}")
            file(CREATE_LINK "${TORCH_LIB_DIR}/${name}" "${TORCH_ROCM_ALIAS_DIR}/${CMAKE_MATCH_1}" SYMBOLIC)
            message(STATUS "Using Torch's ${name} as ${CMAKE_MATCH_1}")
        endif()
    endif()
endforeach()

set(HETEROMEGA_HIP_OPTIONS $<$<COMPILE_LANGUAGE:HIP>:-O3> $<$<COMPILE_LANGUAGE:HIP>:-ffast-math>)

# HeteroMosaic packed W4A16 GEMM and GEMV, built from its sources (no NPU dependencies).
set(HM_SOURCES
    ${HETEROMOSAIC_ROOT}/src/hipkernels/w4a16_gemm_packed.cu
    ${HETEROMOSAIC_ROOT}/src/hipkernels/w4a16_gemv_packed.cu)
set_source_files_properties(${HM_SOURCES} PROPERTIES LANGUAGE HIP)
add_library(hm_w4a16 STATIC ${HM_SOURCES})
set_target_properties(hm_w4a16 PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(hm_w4a16 PUBLIC ${HETEROMOSAIC_ROOT}/include ${ROCM_PATH}/include ${TORCH_INCLUDE_DIRS})
target_compile_definitions(hm_w4a16 PRIVATE __HIP_PLATFORM_AMD__=1 RDNA3=1)
if(HETEROMEGA_GEMM_AUTOTUNE)
    target_compile_definitions(hm_w4a16 PRIVATE AUTOTUNE=1)
    message(STATUS "HeteroMosaic GEMM: AUTOTUNE=1")
endif()
target_compile_options(hm_w4a16 PRIVATE ${HETEROMEGA_HIP_OPTIONS})
target_link_libraries(hm_w4a16 PRIVATE hip::device "${TORCH_LIBRARIES}")
target_link_libraries(hm_w4a16 PUBLIC hip::host)

# HeteroMega GPU library.
set(HETEROMEGA_HIP_SOURCES src/silu_mul.cu src/residual_add.cu src/gemv_w4a16.cu)
set_source_files_properties(${HETEROMEGA_HIP_SOURCES} PROPERTIES LANGUAGE HIP)
add_library(heteromega_gpu STATIC
    ${HETEROMEGA_HIP_SOURCES}
    src/gpu_ffn.cpp
    src/pack_w4a16.cpp
    src/residual_add_cpu.cpp)
set_target_properties(heteromega_gpu PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(heteromega_gpu PUBLIC include ${ROCM_PATH}/include ${TORCH_INCLUDE_DIRS})
target_compile_options(heteromega_gpu PRIVATE ${HETEROMEGA_HIP_OPTIONS} $<$<COMPILE_LANGUAGE:CXX>:-O3>)
target_compile_definitions(heteromega_gpu PUBLIC __HIP_PLATFORM_AMD__=1)
target_link_libraries(heteromega_gpu PUBLIC hm_w4a16 "${TORCH_LIBRARIES}" hip::host)
if(OpenMP_CXX_FOUND)
    target_link_libraries(heteromega_gpu PRIVATE OpenMP::OpenMP_CXX)
endif()

# Tests and benchmarks.
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
foreach(name tests/test_gpu_ffn bench/bench_gpu_ffn_slice bench/bench_ffn_ablation bench/bench_gemv bench/bench_reduce)
    get_filename_component(target ${name} NAME)
    add_executable(${target} ${name}.cpp)
    target_link_libraries(${target} PRIVATE heteromega_gpu Threads::Threads)
    if(EXISTS "${TORCH_ROCM_ALIAS_DIR}")
        set_target_properties(${target} PROPERTIES SKIP_BUILD_RPATH ON)
        target_link_options(${target} PRIVATE "LINKER:--disable-new-dtags" "LINKER:-rpath,${TORCH_ROCM_ALIAS_DIR}:${TORCH_LIB_DIR}")
    endif()
endforeach()

set_source_files_properties(bench/bench_bandwidth.cpp PROPERTIES LANGUAGE HIP)
add_executable(bench_bandwidth bench/bench_bandwidth.cpp)
target_compile_options(bench_bandwidth PRIVATE ${HETEROMEGA_HIP_OPTIONS})
target_link_libraries(bench_bandwidth PRIVATE hip::device Threads::Threads)

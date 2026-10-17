add_library(unified_llm_w4a16_hetero_lib STATIC
    unified_llm_w4a16.cpp
    npuSetup.cpp
    hetero_compute.cpp
    helper.cpp
    ../cpu_avx_kernels/w4a16_gemv_avx_packed.cpp
    ../cpu_avx_kernels/w4a16_gemv_avx_unpacked.cpp
)

if(NOT DEFINED ROCM_PATH)
    set(ROCM_PATH "/opt/rocm-7.1.1")
endif()
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -I${ROCM_PATH}/include")
set(ENV{ROCM_PATH} "${ROCM_PATH}")

find_package(hip REQUIRED)
message(STATUS "Using ROCM_PATH: ${ROCM_PATH}")

target_include_directories(unified_llm_w4a16_hetero_lib PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
    ${ROCM_PATH}/include
    ${ROCM_PATH}/include/hip
)

find_library(HSA_RUNTIME_LIB hsa-runtime64 PATHS ${ROCM_PATH}/lib NO_DEFAULT_PATH)
if(NOT HSA_RUNTIME_LIB)
    message(FATAL_ERROR "hsa-runtime64 not found in ${ROCM_PATH}/lib")
endif()

target_link_libraries(unified_llm_w4a16_hetero_lib PUBLIC
    "${TORCH_LIBRARIES}"
    npu_matmul_func
    hipkernels
    ${HSA_RUNTIME_LIB}
    amdhip64
)

target_compile_definitions(unified_llm_w4a16_hetero_lib PUBLIC
    __HIP_PLATFORM_AMD__=1
)

target_compile_options(unified_llm_w4a16_hetero_lib PRIVATE
    -march=native
    -O3
)

find_package(OpenMP REQUIRED)
if(OpenMP_CXX_FOUND)
    target_link_libraries(unified_llm_w4a16_hetero_lib PUBLIC OpenMP::OpenMP_CXX)
endif()

# HeteroMega GPU FFN (kernels/gpu). With -DHETEROMEGA_GPU_FFN=ON, forward_llama3_with_scratch runs the FFN through
# heteromega::GpuFfnModel when the environment variable HETEROMEGA_GPU_FFN=1 is set. A separate library, so that
# this library's -march=native does not reach the HIP device compiler.
option(HETEROMEGA_GPU_FFN "Build the HeteroMega GPU FFN path into the model library" OFF)
if(HETEROMEGA_GPU_FFN)
    set(HM_GPU_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../kernels/gpu)
    set(HM_GPU_HIP_SOURCES ${HM_GPU_DIR}/src/silu_mul.cu ${HM_GPU_DIR}/src/residual_add.cu ${HM_GPU_DIR}/src/gemv_w4a16.cu)
    set_source_files_properties(${HM_GPU_HIP_SOURCES} PROPERTIES LANGUAGE HIP)
    add_library(heteromega_gpu_ffn STATIC
        ${HM_GPU_HIP_SOURCES}
        ${HM_GPU_DIR}/src/gpu_ffn.cpp
        ${HM_GPU_DIR}/src/pack_w4a16.cpp
        ${HM_GPU_DIR}/src/residual_add_cpu.cpp)
    set_target_properties(heteromega_gpu_ffn PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON POSITION_INDEPENDENT_CODE ON)
    target_include_directories(heteromega_gpu_ffn PUBLIC ${HM_GPU_DIR}/include)
    target_compile_options(heteromega_gpu_ffn PRIVATE
        $<$<COMPILE_LANGUAGE:HIP>:-O3> $<$<COMPILE_LANGUAGE:HIP>:-ffast-math> $<$<COMPILE_LANGUAGE:CXX>:-O3>)
    target_compile_definitions(heteromega_gpu_ffn PUBLIC __HIP_PLATFORM_AMD__=1)
    target_link_libraries(heteromega_gpu_ffn PUBLIC hipkernels "${TORCH_LIBRARIES}" hip::host OpenMP::OpenMP_CXX)
    target_link_libraries(unified_llm_w4a16_hetero_lib PUBLIC heteromega_gpu_ffn)
    target_compile_definitions(unified_llm_w4a16_hetero_lib PRIVATE HETEROMEGA_GPU_FFN=1)
    message(STATUS "HeteroMega GPU FFN: ON (runtime switch HETEROMEGA_GPU_FFN=1)")
endif()

set_target_properties(unified_llm_w4a16_hetero_lib PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    POSITION_INDEPENDENT_CODE ON
)

# Python bindings were merged into this directory from unified_llm_w4a16_hetero_pybind.
find_package(pybind11 REQUIRED)
find_package(Python COMPONENTS Interpreter Development REQUIRED)

execute_process(
    COMMAND ${Python_EXECUTABLE} -c "import torch; import os; print(os.path.dirname(torch.__file__))"
    OUTPUT_VARIABLE TORCH_PYTHON_PATH
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

if(TORCH_PYTHON_PATH AND EXISTS "${TORCH_PYTHON_PATH}/share/cmake/Torch")
    set(CMAKE_PREFIX_PATH "${TORCH_PYTHON_PATH}/share/cmake/Torch" ${CMAKE_PREFIX_PATH})
    message(STATUS "Found PyTorch in Python environment: ${TORCH_PYTHON_PATH}")
endif()

find_package(Torch QUIET)
if(NOT Torch_FOUND)
    message(STATUS "PyTorch not found in Python environment, using standalone LibTorch")
endif()

if(CMAKE_CXX_FLAGS)
    string(REPLACE "-Wno-duplicate-decl-specifier" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
endif()
if(TORCH_CXX_FLAGS)
    string(REPLACE "-Wno-duplicate-decl-specifier" "" TORCH_CXX_FLAGS "${TORCH_CXX_FLAGS}")
endif()

get_target_property(PYBIND11_INCLUDE_DIR pybind11::headers INTERFACE_INCLUDE_DIRECTORIES)
if(NOT PYBIND11_INCLUDE_DIR)
    get_target_property(PYBIND11_INCLUDE_DIR pybind11::pybind11 INTERFACE_INCLUDE_DIRECTORIES)
endif()
if(NOT PYBIND11_INCLUDE_DIR)
    set(PYBIND11_INCLUDE_DIR ${Python_INCLUDE_DIRS})
endif()

add_library(unified_llm_w4a16_hetero_libtorch MODULE
    python_bindings.cpp
)

set_target_properties(unified_llm_w4a16_hetero_libtorch PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    PREFIX ""
    OUTPUT_NAME "unified_llm_w4a16_hetero_libtorch"
)

target_include_directories(unified_llm_w4a16_hetero_libtorch PRIVATE
    ${PYBIND11_INCLUDE_DIR}
    ${Python_INCLUDE_DIRS}
    ${DRIVER_PATH}/src/include/uapi/drm_local
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
)

if(TORCH_PYTHON_PATH)
    if(EXISTS "${TORCH_PYTHON_PATH}/include")
        target_include_directories(unified_llm_w4a16_hetero_libtorch PRIVATE "${TORCH_PYTHON_PATH}/include")
    endif()
    if(EXISTS "${TORCH_PYTHON_PATH}/include/torch/csrc/api/include")
        target_include_directories(unified_llm_w4a16_hetero_libtorch PRIVATE "${TORCH_PYTHON_PATH}/include/torch/csrc/api/include")
    endif()
endif()

if(TORCH_PYTHON_PATH AND EXISTS "${TORCH_PYTHON_PATH}/lib")
    link_directories("${TORCH_PYTHON_PATH}/lib")
endif()
link_directories(${ROCM_PATH}/lib)

target_link_libraries(unified_llm_w4a16_hetero_libtorch PRIVATE
    unified_llm_w4a16_hetero_lib
    "${TORCH_LIBRARIES}"
    pybind11::module
    pybind11::windows_extras
    npu_matmul_func
    ${HSA_RUNTIME_LIB}
)

if(TORCH_PYTHON_PATH AND EXISTS "${TORCH_PYTHON_PATH}/lib")
    find_library(TORCH_PYTHON_LIBRARY torch_python PATHS "${TORCH_PYTHON_PATH}/lib" NO_DEFAULT_PATH)
    if(TORCH_PYTHON_LIBRARY)
        target_link_libraries(unified_llm_w4a16_hetero_libtorch PRIVATE ${TORCH_PYTHON_LIBRARY})
        message(STATUS "Found torch_python library: ${TORCH_PYTHON_LIBRARY}")
    else()
        message(WARNING "torch_python library not found in ${TORCH_PYTHON_PATH}/lib")
    endif()
endif()

if(TORCH_PYTHON_PATH AND EXISTS "${TORCH_PYTHON_PATH}/lib")
    set_target_properties(unified_llm_w4a16_hetero_libtorch PROPERTIES
        INSTALL_RPATH "${TORCH_PYTHON_PATH}/lib"
        BUILD_RPATH "${TORCH_PYTHON_PATH}/lib"
        INSTALL_RPATH_USE_LINK_PATH TRUE
    )
else()
    set_target_properties(unified_llm_w4a16_hetero_libtorch PROPERTIES
        INSTALL_RPATH_USE_LINK_PATH TRUE
    )
endif()

target_compile_definitions(unified_llm_w4a16_hetero_libtorch PRIVATE
    VERSION_INFO=${pybind11_VERSION}
    __HIP_PLATFORM_AMD__=1
)

file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/py/unified_llm_w4a16")
set_target_properties(unified_llm_w4a16_hetero_libtorch PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/py/unified_llm_w4a16"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/py/unified_llm_w4a16"
)

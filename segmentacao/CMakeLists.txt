cmake_minimum_required(VERSION 2.6)
project(segmentacao_sequencial)

set (CMAKE_CXX_STANDARD 11)

find_package(CUDA QUIET REQUIRED)

# Pass options to NVCC
set(
    CUDA_NVCC_FLAGS
    ${CUDA_NVCC_FLAGS};
    -O3 -gencode arch=compute_22,code=sm_22
    )

# Specify include directories
include_directories(
    kernels
    utility
    )

# Specify library paths
link_directories(
    /opt/foobar/lib
    /opt/joestuff/lib
)

cuda_add_executable(
    segmentacao_cuda
    main_cuda.cu
    imagem.cpp
    )

# For linking ...
# Specify target & libraries to link it with
target_link_libraries(
    segmentacao_cuda
    -lfoobar
    -ljoestuff
    )

add_executable(segmentacao_sequencial main_seq.cpp imagem.cpp)

install(TARGETS segmentacao_sequencial RUNTIME DESTINATION bin)

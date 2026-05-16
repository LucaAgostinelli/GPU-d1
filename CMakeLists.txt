cmake_minimum_required(VERSION 3.20)

project(cuda_project LANGUAGES C CXX CUDA)

# =========================
# Language standards
# =========================
set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 14)

set(CMAKE_CUDA_ARCHITECTURES 80) # AMPERE!

# =========================
# Output folders
# =========================
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/build)

# =========================
# Source files
# =========================
set(SOURCES
    src/main.cu
    src/matrix.cu
    src/utils.cu
    src/cpu.cu
    src/spMVflat.cu
    src/spMVline.cu
    src/spMVacc.cu
    src/spMVsv-adaptive.cu
    src/spMVell.cu
    src/spMVhyb.cu
    src/spMVscalar.cu
    src/spMVcusparse.cu
)

# =========================
# Executable
# =========================
add_executable(spmv ${SOURCES})

# =========================
# Include directories
# =========================
target_include_directories(spmv PRIVATE include)

# =========================
# Link libraries
# =========================
target_link_libraries(spmv m)
target_link_libraries(spmv cusparse)

# =========================
# CUDA specific options
# =========================
set_target_properties(spmv PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
)

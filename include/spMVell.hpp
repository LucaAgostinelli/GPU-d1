#pragma once
#include <cuda_runtime.h>
#include "matrix.hpp"
#include "utils.hpp"

// ============================================================
// ELL SpMV kernel  (Bell & Garland 2009)
// ============================================================
//
// One thread per row.
// Each thread iterates over the max_col columns of its ELL row,
// accumulating sum += values[c*nrows+row] * x[col_idx[c*nrows+row]]
//
// Coalescing: warp of 32 threads reads 32 consecutive elements
//   values[c*nrows + row_base .. row_base+31]
// --> one 128-byte cache line per column-slot iteration.
//
// No shared memory, no atomics, no synchronisation needed.
//
__global__ void spmv_ell_kernel(int nrows,
                                int max_col,
                                const int *__restrict__ col_idx,
                                const float *__restrict__ values,
                                const float *__restrict__ x,
                                float *__restrict__ y);

// The fill_ratio_threshold parameter guards against pathological matrices:
// if the ELL fill ratio (real NNZ / allocated slots) is below the threshold
// the launcher prints a warning but still runs
KernelResults spmv_ell_launcher(const CSRHost &h_csr,
                                const float *d_x,
                                float *d_y,
                                float peak_bw_gbs,
                                float fill_ratio_threshold = 0.5f);
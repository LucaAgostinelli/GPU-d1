#pragma once
#include <cuda_runtime.h>
#include "utils.hpp"
#include "matrix.hpp"

// ============================================================
// LINE-ENHANCE METHOD  (Chu et al., 2024)
// ============================================================
//
// Key idea: hybrid row-splitting + NNZ-splitting.
//   - Each workgroup (block) owns exactly N = rows_per_wg complete rows
//     --> no row ever spans two workgroups --> no atomicAdd on y[]
//   - Within the workgroup, non-zeros are split equally among threads
//     using multiple rounds (each round loads R x THREADS NNZs into
//     shared memory).
//   - Reduction is done by dividing the THREADS threads of the block
//     into V vectors (V is a power of 2); each vector reduces one row
//     using a parallel prefix sum, then thread 0 of the vector writes
//     the result directly to y[].
//
// Parameters (fixed, matching paper's recommended defaults):
//   THREADS = 512   threads per block
//   R       = 2     NNZs loaded per thread per round  (--> shared = R×THREADS floats)
//   V       = 1     threads per reduction vector (adaptive: 1, 2, 4,...)
//                   (chosen at launch time based on avg NNZ/row)

KernelResults spmv_line_launcher(const CSRDevice &d_csr,
                                 const float *d_x,
                                 float *d_y);

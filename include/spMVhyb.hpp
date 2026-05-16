#pragma once
#include <cuda_runtime.h>
#include "utils.hpp"
#include "matrix.hpp"

// ============================================================
// HYB (Hybrid ELL + COO) SpMV  (Bell & Garland, 2009)
// ============================================================
//
// Idea: store the first K nonzeros of each row in a padded ELL
// structure (coalesced, branch-free), and overflow the remaining
// nonzeros of the "long" rows into a COO tail (handled by a
// separate segmented-reduction kernel).
//
// Choosing K (the ELL "width"):
//   Bell & Garland model ELL as ~3× faster than COO when the ELL
//   portion is fully utilised. Under that model it is profitable
//   to add a K-th column to the ELL part when the number of rows
//   with at least K NNZs is >= max(4096, nrows/3).
//
// ELL kernel - one thread per row, column-major layout.
// COO kernel - one thread per nonzero; partial sums for the same
//              row are accumulated with atomicAdd (the number of
//              COO entries is small by construction).

KernelResults spmv_hyb_launcher(const CSRHost &h_csr,
                                const float *d_x,
                                float *d_y);

#pragma once
#include "matrix.hpp"
#include "utils.hpp"

// test_mode=false: runs only the heuristic-chosen kernel
// test_mode=true: runs both kernels, compares outputs and timing,
//                 validates the heuristic, then copies the chosen result to d_y.
KernelResults spmv_acc_launcher(const CSRDevice &d_csr,
                                const CSRHost &h_csr,
                                const float *d_x,
                                float *d_y,
                                bool test_mode = false);
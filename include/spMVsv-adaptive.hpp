#pragma once
#include <cuda_runtime.h>
#include "utils.hpp"
#include "matrix.hpp"

KernelResults spmv_sv_adaptive_launcher(const CSRDevice &d_csr,
                                   const CSRHost &h_csr,
                                   const float *d_x,
                                   float *d_y);

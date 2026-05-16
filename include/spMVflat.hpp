#pragma once
#include <cuda_runtime.h>
#include <utils.hpp>
#include <matrix.hpp>

KernelResults spmv_flat_launcher(const CSRDevice &d_csr,
                                 const float *d_x,
                                 float *d_y);

#pragma once
#include "matrix.hpp"
#include "utils.hpp"

KernelResults spmv_scalar_launcher(const CSRDevice &d_csr,
                                  const float *d_x,
                                  float *d_y);
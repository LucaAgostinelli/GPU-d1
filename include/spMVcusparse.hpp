#pragma once
#include <cuda_runtime.h>
#include <cusparse.h>
#include "utils.hpp"
#include "matrix.hpp"

KernelResults spmv_cusparse_launcher(const CSRDevice &d_csr,
                                     const float *d_x,
                                     float *d_y);

#pragma once
#include "matrix.hpp"
#include "utils.hpp"

void spmv_cpu(int nrows,
              const CSRHost &csr,
              const std::vector<float> &x,
              std::vector<float> &y);

void benchmark_cpu_spmv(int nrows,
                        const CSRHost &csr,
                        const std::vector<float> &h_x,
                        std::vector<float> &h_y_cpu);

bool almost_equal(float a, float b, float abs_tol = 1e-2f, float rel_tol = 1e-2f);

void check_correctness(int nrows,
                       const std::vector<float> &h_y_cpu,
                       const std::vector<float> &h_y_gpu);

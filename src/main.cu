#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "matrix.hpp"
#include "utils.hpp"
#include "spMVflat.hpp"
#include "spMVline.hpp"
#include "spMVacc.hpp"
#include "spMVsv-adaptive.hpp"
#include "spMVell.hpp"
#include "spMVhyb.hpp"
#include "spMVscalar.hpp"
#include "spMVcusparse.hpp"
#include "cpu.hpp"

// Valid kernel names (also used in sbatchOne.sh --array mapping)
// ALL, SCALAR, S-V-ADAPTIVE, LINE, FLAT, ACC, ELL, HYB, cuSPARSE
int main(int argc, char **argv)
{
    int nrows, ncols;
    bool symmetric;

    if (argc < 2)
    {
        fprintf(stderr, "Usage: %s <matrix_path> [kernel]\n", argv[0]);
        fprintf(stderr, "  kernel: ALL (default) | SCALAR | S-V-ADAPTIVE | LINE | FLAT | ACC | ELL | HYB | cuSPARSE\n");
        return 1;
    }

    std::string path = argv[1];
    std::string kernel_filter = (argc >= 3) ? argv[2] : "ALL";

    auto run = [&](const std::string &name) -> bool
    {
        return kernel_filter == "ALL" || kernel_filter == name;
    };

    // ----------------------------------------------------------
    // Matrix parsing
    // ----------------------------------------------------------
    auto coo = read_mtx(path, nrows, ncols, symmetric);
    CSRHost h_csr = coo_to_csr(coo, nrows, ncols);
    CSRDevice d_csr = csr_host_to_device(h_csr);

    // ----------------------------------------------------------
    // Allocate vectors
    // ----------------------------------------------------------
    float *d_x;
    float *d_y;

    cudaMalloc(&d_x, ncols * sizeof(float));
    cudaMalloc(&d_y, nrows * sizeof(float));

    std::vector<float> h_x = generateRandomArray(ncols);

    cudaMemcpy(d_x, h_x.data(), ncols * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_y, 0, nrows * sizeof(float));

    // ----------------------------------------------------------
    // Peak bandwidth
    // ----------------------------------------------------------
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    float peak_bw_gbs = 2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6;

    // ----------------------------------------------------------
    // CPU reference
    // ----------------------------------------------------------
    std::vector<float> h_y_cpu(nrows, 0.0f);
    benchmark_cpu_spmv(nrows, h_csr, h_x, h_y_cpu);

    // ----------------------------------------------------------
    // Run kernels
    // ----------------------------------------------------------

    auto gflops = [&](const KernelResults &r) -> double
    {
        return 2.0 * h_csr.nnz / (r.avg_ms / 1e3) / 1e9;
    };

    auto csv_row = [&](const char *kname, const KernelResults &r)
    {
        double gf = gflops(r);
        double bw = r.eff_bw_gbs;
        double pct = 100.0 * bw / peak_bw_gbs;
        printf("RESULT_CSV,%s,%d,%d,%d,%s,%.6f,%.6f,%.6f,%.9f,%.6f,%.4f,%.2f\n",
               path.c_str(), nrows, ncols, h_csr.nnz, kname,
               r.avg_ms, r.min_ms, r.max_ms, r.variance_ms,
               gf, bw, pct);
    };

    if (run("SCALAR"))
    {
        printf("---------------- running SCALAR ----------------\n");
        CUDA_CHECK(cudaMemset(d_y, 0, nrows * sizeof(float)));
        KernelResults r = spmv_scalar_launcher(d_csr, d_x, d_y);
        r.report(peak_bw_gbs);
        std::vector<float> h_y(nrows);
        CUDA_CHECK(cudaMemcpy(h_y.data(), d_y, nrows * sizeof(float), cudaMemcpyDeviceToHost));
        check_correctness(nrows, h_y_cpu, h_y);
        csv_row("SCALAR", r);
    }

    if (run("S-V-ADAPTIVE"))
    {
        printf("---------------- running S-V-ADAPTIVE ----------------\n");
        CUDA_CHECK(cudaMemset(d_y, 0, nrows * sizeof(float)));
        KernelResults r = spmv_sv_adaptive_launcher(d_csr, h_csr, d_x, d_y);
        r.report(peak_bw_gbs);
        std::vector<float> h_y(nrows);
        CUDA_CHECK(cudaMemcpy(h_y.data(), d_y, nrows * sizeof(float), cudaMemcpyDeviceToHost));
        check_correctness(nrows, h_y_cpu, h_y);
        csv_row("S-V-ADAPTIVE", r);
    }

    if (run("LINE"))
    {
        printf("---------------- running LINE ----------------\n");
        CUDA_CHECK(cudaMemset(d_y, 0, nrows * sizeof(float)));
        KernelResults r = spmv_line_launcher(d_csr, d_x, d_y);
        r.report(peak_bw_gbs);
        std::vector<float> h_y(nrows);
        CUDA_CHECK(cudaMemcpy(h_y.data(), d_y, nrows * sizeof(float), cudaMemcpyDeviceToHost));
        check_correctness(nrows, h_y_cpu, h_y);
        csv_row("LINE", r);
    }

    if (run("FLAT"))
    {
        printf("---------------- running FLAT ----------------\n");
        CUDA_CHECK(cudaMemset(d_y, 0, nrows * sizeof(float)));
        KernelResults r = spmv_flat_launcher(d_csr, d_x, d_y);
        r.report(peak_bw_gbs);
        std::vector<float> h_y(nrows);
        CUDA_CHECK(cudaMemcpy(h_y.data(), d_y, nrows * sizeof(float), cudaMemcpyDeviceToHost));
        check_correctness(nrows, h_y_cpu, h_y);
        csv_row("FLAT", r);
    }

    if (run("ACC"))
    {
        printf("---------------- running ACC ----------------\n");
        CUDA_CHECK(cudaMemset(d_y, 0, nrows * sizeof(float)));
        KernelResults r = spmv_acc_launcher(d_csr, h_csr, d_x, d_y, false);
        r.report(peak_bw_gbs);
        std::vector<float> h_y(nrows);
        CUDA_CHECK(cudaMemcpy(h_y.data(), d_y, nrows * sizeof(float), cudaMemcpyDeviceToHost));
        check_correctness(nrows, h_y_cpu, h_y);
        csv_row("ACC", r);
    }

    if (run("ELL"))
    {
        printf("---------------- running ELL ----------------\n");
        CUDA_CHECK(cudaMemset(d_y, 0, nrows * sizeof(float)));
        KernelResults r = spmv_ell_launcher(h_csr, d_x, d_y, peak_bw_gbs, 0.5f);
        r.report(peak_bw_gbs);
        if (r.avg_ms >= 0.0f)
        {
            std::vector<float> h_y(nrows);
            CUDA_CHECK(cudaMemcpy(h_y.data(), d_y, nrows * sizeof(float), cudaMemcpyDeviceToHost));
            check_correctness(nrows, h_y_cpu, h_y);
        }
        if (r.avg_ms >= 0.0f)
            csv_row("ELL", r);
    }

    if (run("HYB"))
    {
        printf("---------------- running HYB ----------------\n");
        CUDA_CHECK(cudaMemset(d_y, 0, nrows * sizeof(float)));
        KernelResults r = spmv_hyb_launcher(h_csr, d_x, d_y);
        r.report(peak_bw_gbs);
        std::vector<float> h_y(nrows);
        CUDA_CHECK(cudaMemcpy(h_y.data(), d_y, nrows * sizeof(float), cudaMemcpyDeviceToHost));
        check_correctness(nrows, h_y_cpu, h_y);
        csv_row("HYB", r);
    }

    if (run("cuSPARSE"))
    {
        printf("---------------- running cuSPARSE ----------------\n");
        CUDA_CHECK(cudaMemset(d_y, 0, nrows * sizeof(float)));
        KernelResults r = spmv_cusparse_launcher(d_csr, d_x, d_y);
        r.report(peak_bw_gbs);
        std::vector<float> h_y(nrows);
        CUDA_CHECK(cudaMemcpy(h_y.data(), d_y, nrows * sizeof(float), cudaMemcpyDeviceToHost));
        check_correctness(nrows, h_y_cpu, h_y);
        csv_row("cuSPARSE", r);
    }

    cudaFree(d_csr.row_ptr);
    cudaFree(d_csr.col_idx);
    cudaFree(d_csr.values);
    cudaFree(d_x);
    cudaFree(d_y);

    return 0;
}
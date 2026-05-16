#include "spMVhyb.hpp"
#include <algorithm>
#include <vector>
#include <cstdio>
#include "spMVell.hpp"

// ============================================================
// HYB (Hybrid ELL + COO) SpMV (Bell & Garland, 2009)
// ============================================================

// ============================================================
// COO kernel
// ============================================================
//
// One thread per COO nonzero.  Because the COO tail is (by design)
// small and irregular, I use atomicAdd on y[]. Coalesced access
// on coo_row/coo_col/coo_val.
__global__ void spmv_hyb_coo(int coo_nnz,
                             const int *__restrict__ coo_row,
                             const int *__restrict__ coo_col,
                             const float *__restrict__ coo_val,
                             const float *__restrict__ x,
                             float *__restrict__ y)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= coo_nnz)
        return;

    int row = __ldg(&coo_row[tid]);
    int col = __ldg(&coo_col[tid]);
    float val = __ldg(&coo_val[tid]);

    atomicAdd(&y[row], val * __ldg(&x[col]));
}

KernelResults spmv_hyb_launcher(const CSRHost &h_csr,
                                const float *d_x,
                                float *d_y)
{
    int nrows = h_csr.nrows;
    int ncols = h_csr.ncols;
    int nnz = h_csr.nnz;
    int ell_cols = 0;
    HYBHost h_hyb = csr_to_hyb(h_csr, ell_cols);

    const size_t ell_total = (size_t)ell_cols * nrows;
    const size_t ell_mem_mb = ell_total * (sizeof(int) + sizeof(float)) / (1024 * 1024);
    const int coo_nnz = h_hyb.coo_nnz;
    const int ell_nnz = nnz - coo_nnz;

    printf("GPU SpMV (HYB - ELL + COO):\n");
    printf("  ELL width K:            %d\n", ell_cols);
    printf("  ELL total slots:        %zu\n", ell_total);
    printf("  ELL memory (col+val):   %zu MB\n", ell_mem_mb);
    printf("  ELL NNZs (real):        %d  (%.1f%%)\n",
           ell_nnz, 100.0f * ell_nnz / std::max(1, nnz));
    printf("  COO NNZs (overflow):    %d  (%.1f%%)\n",
           coo_nnz, 100.0f * coo_nnz / std::max(1, nnz));
    if (ell_total > 0)
        printf("  ELL fill ratio:         %.3f\n",
               (float)ell_nnz / (float)ell_total);

    HYBDevice d_hyb = hyb_host_to_device(h_hyb);

    const int BLOCK_ELL = 256;
    const int GRID_ELL = (nrows + BLOCK_ELL - 1) / BLOCK_ELL;

    const int BLOCK_COO = 256;
    const int GRID_COO = (coo_nnz > 0) ? (coo_nnz + BLOCK_COO - 1) / BLOCK_COO : 0;

    // Helper: launch both ELL and COO kernels
    auto launch_both = [&]()
    {
        // ELL pass: initialises y[]
        spmv_ell_kernel<<<GRID_ELL, BLOCK_ELL>>>(
            nrows, ell_cols,
            d_hyb.ell_col_idx, d_hyb.ell_values,
            d_x, d_y);

        // COO pass: accumulates overflow into y[] (atomicAdd)
        if (coo_nnz > 0 && GRID_COO > 0)
            spmv_hyb_coo<<<GRID_COO, BLOCK_COO>>>(
                coo_nnz,
                d_hyb.coo_row, d_hyb.coo_col, d_hyb.coo_val,
                d_x, d_y);
    };

    // Warm-up
    for (int i = 0; i < WARMUP_ITERATIONS; i++)
    {
        CUDA_CHECK(cudaMemset(d_y, 0, nrows * sizeof(float)));
        launch_both();
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // Benchmark
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    float total_ms = 0.0f, min_ms = 1e30f, max_ms = 0.0f;
    float total_ms2 = 0.0f;
    for (int i = 0; i < BENCHMARK_ITERATIONS; i++)
    {
        CUDA_CHECK(cudaMemset(d_y, 0, nrows * sizeof(float)));

        CUDA_CHECK(cudaEventRecord(start));
        launch_both();
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        total_ms += ms;
        total_ms2 += ms * ms;
        min_ms = fminf(min_ms, ms);
        max_ms = fmaxf(max_ms, ms);
    }

    printf("\n  [Kernel]\n");
    printf("  ELL: BLOCK=%d GRID=%d\n", BLOCK_ELL, GRID_ELL);
    printf("  COO: BLOCK=%d GRID=%d\n", BLOCK_COO, GRID_COO);
    printf("\n");

    float avg_ms = total_ms / BENCHMARK_ITERATIONS;
    float variance_ms = total_ms2 / BENCHMARK_ITERATIONS - avg_ms * avg_ms;

    // Padded: ELL slots (col+val) + COO arrays + x + y
    long long bytes_padded =
        (long long)ell_total * sizeof(float)      // ell_values[]: all slots (incl. padding)
        + (long long)ell_total * sizeof(int)      // ell_col_idx[]: all slots (incl. padding)
        + (long long)coo_nnz * sizeof(float)      // coo_val[]
        + (long long)coo_nnz * sizeof(int)        // coo_col[]
        + (long long)coo_nnz * sizeof(int)        // coo_row[]
        + (long long)ncols * sizeof(float)        // x[]: upper bound (both kernels gather from x)
        + (long long)nrows * sizeof(float)        // y[]: ELL writes it once
        + (long long)coo_nnz * sizeof(float) * 2; // y[]: COO atomicAdd = 1 read + 1 write per entry

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    hyb_device_free(d_hyb);
    CUDA_CHECK(cudaGetLastError());

    return KernelResults("HYBRID", avg_ms, min_ms, max_ms, variance_ms, bytes_padded);
}

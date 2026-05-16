#include "spMVell.hpp"

// ============================================================
// ELL SpMV kernel
// ============================================================
//
// Grid / block layout
//   blockDim.x = BLOCK  (e.g. 256)
//   gridDim.x  = ceil(nrows / BLOCK)
//   --> one thread per row
//
// Memory access pattern (column-major ELL):
//   For column slot c, all threads in a warp read:
//     col_idx[ c*nrows + row_base ],  col_idx[ c*nrows + row_base+1 ], ...
//   These are 32 consecutive 4-byte integers at address
//     &col_idx[c*nrows + row_base]
//   --> single 128-byte transaction, fully coalesced.
//   Same for values[].
//
// Padding rows (values == 0) contribute 0 to the sum automatically,
// so no branch is needed to skip them.
//
// The __ldg() intrinsic routes reads through the read-only texture
// cache (L1 texture), which is beneficial when the same x[] element
// is loaded by many threads.
//
__global__ void spmv_ell_kernel(int nrows,
                                int max_col,
                                const int *__restrict__ col_idx,
                                const float *__restrict__ values,
                                const float *__restrict__ x,
                                float *__restrict__ y)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= nrows)
        return;

    float sum = 0.0f;
    for (int c = 0; c < max_col; c++)
    {
        int idx = c * nrows + row;
        float val = __ldg(&values[idx]);
        int col = __ldg(&col_idx[idx]);
        sum += val * __ldg(&x[col]);
    }

    y[row] = sum;
}

KernelResults spmv_ell_launcher(const CSRHost &h_csr,
                                const float *d_x,
                                float *d_y,
                                float peak_bw_gbs,
                                float fill_ratio_threshold)
{
    float fill_ratio = 0.0f;
    ELLHost h_ell = csr_to_ell(h_csr, fill_ratio);

    // fill_ratio == -1 is the sentinel set by csr_to_ell when the matrix
    // would exceed MAX_ELL_BYTES. The struct is empty; skip everything.
    if (fill_ratio < 0.0f)
    {
        printf("GPU SpMV (ELL):\n");
        printf("  [ELL] SKIPPED - max_col * nrows would exceed memory limit.\n");
        printf("  This matrix has a highly skewed NNZ distribution (power-law).\n");
        printf("  Use HYB (ELL + COO) instead.\n\n");
        return KernelResults("ELL", -1.0f, -1.0f, -1.0f, -1.0f, -1.0f);
    }

    const long long ell_total_slots = (long long)h_ell.max_col * h_ell.nrows;
    const double ell_mem_mb = ell_total_slots * (sizeof(int) + sizeof(float)) / 1e6;

    printf("GPU SpMV (ELL):\n");
    printf("  max_col (padded cols):   %d\n", h_ell.max_col);
    printf("  ELL total slots:         %lld\n", ell_total_slots);
    printf("  ELL memory (col+val):    %.1f MB\n", ell_mem_mb);
    printf("  fill ratio:              %.3f  (%s)\n",
           fill_ratio,
           fill_ratio >= fill_ratio_threshold ? "OK" : "WARNING: low fill, ELL wastes memory");

    if (fill_ratio < fill_ratio_threshold)
        printf("  [ELL] fill ratio %.3f < threshold %.3f - consider HYB instead\n",
               fill_ratio, fill_ratio_threshold);

    ELLDevice d_ell = ell_host_to_device(h_ell);

    const int BLOCK = 256;
    const int GRID = (h_ell.nrows + BLOCK - 1) / BLOCK;

    // Warm-up
    for (int i = 0; i < WARMUP_ITERATIONS; i++)
    {
        CUDA_CHECK(cudaMemset(d_y, 0, h_ell.nrows * sizeof(float)));
        spmv_ell_kernel<<<GRID, BLOCK>>>(
            h_ell.nrows, h_ell.max_col,
            d_ell.col_idx, d_ell.values,
            d_x, d_y);
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
        CUDA_CHECK(cudaMemset(d_y, 0, h_ell.nrows * sizeof(float)));

        CUDA_CHECK(cudaEventRecord(start));
        spmv_ell_kernel<<<GRID, BLOCK>>>(
            h_ell.nrows, h_ell.max_col,
            d_ell.col_idx, d_ell.values,
            d_x, d_y);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        total_ms += ms;
        total_ms2 += ms * ms;
        min_ms = fminf(min_ms, ms);
        max_ms = fmaxf(max_ms, ms);
    }

    // The ELL kernel reads the PADDED array, not only the real NNZ:
    // max_col * nrows slots are all read, including padding

    float avg_ms = total_ms / BENCHMARK_ITERATIONS;
    float variance_ms = total_ms2 / BENCHMARK_ITERATIONS - avg_ms * avg_ms;
    long long bytes =
        ell_total_slots * sizeof(float)           // values[]: all ELL slots (padding = 0.0f, still read)
        + ell_total_slots * sizeof(int)           // col_idx[]: all ELL slots (padding --> col 0, still read)
        + (long long)h_ell.ncols * sizeof(float)  // x[]: upper bound = all columns
        + (long long)h_ell.nrows * sizeof(float); // y[]: one float written per row

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    ell_device_free(d_ell);
    CUDA_CHECK(cudaGetLastError());

    return KernelResults("ELL", avg_ms, min_ms, max_ms, variance_ms, bytes);
}
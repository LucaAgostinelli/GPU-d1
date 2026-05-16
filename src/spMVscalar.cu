#include "spMVscalar.hpp"

__global__ void spmv_scalar(int nrows,
                            const int *__restrict__ row_ptr,
                            const int *__restrict__ col_idx,
                            const float *__restrict__ values,
                            const float *__restrict__ x,
                            float *__restrict__ y)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < nrows)
    {
        float sum = 0.0f;

        for (int nnz_idx = row_ptr[row]; nnz_idx < row_ptr[row + 1]; nnz_idx++)
        {
            sum += values[nnz_idx] * x[col_idx[nnz_idx]];
        }

        y[row] = sum;
    }
}

KernelResults spmv_scalar_launcher(const CSRDevice &d_csr,
                                   const float *d_x,
                                   float *d_y)
{
    int nrows = d_csr.nrows;
    int ncols = d_csr.ncols;
    int nnz = d_csr.nnz;

    int blockSize = 256;
    int gridSize = (nrows + blockSize - 1) / blockSize;

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // WARM-UP
    for (int i = 0; i < WARMUP_ITERATIONS; i++)
    {
        spmv_scalar<<<gridSize, blockSize>>>(
            nrows,
            d_csr.row_ptr,
            d_csr.col_idx,
            d_csr.values,
            d_x,
            d_y);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    float total_ms = 0.0f;
    float min_ms = 1e30f;
    float max_ms = 0.0f;
    float total_ms2 = 0.0f;

    for (int i = 0; i < BENCHMARK_ITERATIONS; i++)
    {
        cudaMemset(d_y, 0, nrows * sizeof(float));

        CUDA_CHECK(cudaEventRecord(start));

        spmv_scalar<<<gridSize, blockSize>>>(
            nrows,
            d_csr.row_ptr,
            d_csr.col_idx,
            d_csr.values,
            d_x,
            d_y);

        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

        total_ms += ms;
        total_ms2 += ms * ms;
        min_ms = fminf(min_ms, ms);
        max_ms = fmaxf(max_ms, ms);
    }

    float avg_ms = total_ms / BENCHMARK_ITERATIONS;
    float variance_ms = total_ms2 / BENCHMARK_ITERATIONS - avg_ms * avg_ms;
    long long bytes = (long long)nnz * 4           // values
                      + (long long)nnz * 4         // col_idx
                      + (long long)(nrows + 1) * 4 // row_ptr
                      + (long long)ncols * 4       // x
                      + (long long)nrows * 4;      // y write

    KernelResults results = KernelResults("SCALAR", avg_ms, min_ms, max_ms, variance_ms, bytes);

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaGetLastError());

    return results;
}

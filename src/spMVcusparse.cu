#include "spMVcusparse.hpp"
#include <cstdio>
#include <cstdlib>
#include <algorithm>

// cuSPARSE error checking
#define CUSPARSE_CHECK(call)                                \
    do                                                      \
    {                                                       \
        cusparseStatus_t _st = (call);                      \
        if (_st != CUSPARSE_STATUS_SUCCESS)                 \
        {                                                   \
            fprintf(stderr, "cuSPARSE error %d at %s:%d\n", \
                    (int)_st, __FILE__, __LINE__);          \
            exit(EXIT_FAILURE);                             \
        }                                                   \
    } while (0)

KernelResults spmv_cusparse_launcher(const CSRDevice &d_csr,
                                     const float *d_x,
                                     float *d_y)
{
    const int nrows = d_csr.nrows;
    const int ncols = d_csr.ncols;
    const int nnz = d_csr.nnz;

    cusparseHandle_t handle;
    CUSPARSE_CHECK(cusparseCreate(&handle));

    cusparseSpMatDescr_t matA;
    CUSPARSE_CHECK(cusparseCreateCsr(
        &matA,
        nrows, ncols, nnz,
        (void *)d_csr.row_ptr, // row offsets  (int32)
        (void *)d_csr.col_idx, // column indices (int32)
        (void *)d_csr.values,  // values (float32)
        CUSPARSE_INDEX_32I,    // row-ptr index type
        CUSPARSE_INDEX_32I,    // col-idx index type
        CUSPARSE_INDEX_BASE_ZERO,
        CUDA_R_32F));

    cusparseDnVecDescr_t vecX, vecY;
    CUSPARSE_CHECK(cusparseCreateDnVec(&vecX, ncols, (void *)d_x, CUDA_R_32F));
    CUSPARSE_CHECK(cusparseCreateDnVec(&vecY, nrows, (void *)d_y, CUDA_R_32F));

    const float alpha = 1.0f;
    const float beta = 0.0f;

    size_t bufferSize = 0;
    CUSPARSE_CHECK(cusparseSpMV_bufferSize(
        handle,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, matA, vecX,
        &beta, vecY,
        CUDA_R_32F,
        CUSPARSE_SPMV_ALG_DEFAULT,
        &bufferSize));

    void *d_buffer = nullptr;
    if (bufferSize > 0)
        CUDA_CHECK(cudaMalloc(&d_buffer, bufferSize));

    // Warm-up
    for (int i = 0; i < WARMUP_ITERATIONS; i++)
    {
        CUDA_CHECK(cudaMemset(d_y, 0, nrows * sizeof(float)));
        CUSPARSE_CHECK(cusparseSpMV(
            handle,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha, matA, vecX,
            &beta, vecY,
            CUDA_R_32F,
            CUSPARSE_SPMV_ALG_DEFAULT,
            d_buffer));
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
        CUSPARSE_CHECK(cusparseSpMV(
            handle,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha, matA, vecX,
            &beta, vecY,
            CUDA_R_32F,
            CUSPARSE_SPMV_ALG_DEFAULT,
            d_buffer));
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        total_ms += ms;
        total_ms2 += ms * ms;
        min_ms = fminf(min_ms, ms);
        max_ms = fmaxf(max_ms, ms);
    }

    printf("GPU SpMV (cuSPARSE - SPMV_ALG_DEFAULT / CSR):\n");
    printf("  workspace:  %zu bytes\n", bufferSize);
    printf("\n");

    float avg_ms = total_ms / BENCHMARK_ITERATIONS;
    float variance_ms = total_ms2 / BENCHMARK_ITERATIONS - avg_ms * avg_ms;

    const long long bytes =
        (long long)nnz * sizeof(float)         // values[]
        + (long long)nnz * sizeof(int)         // col_idx[]
        + (long long)(nrows + 1) * sizeof(int) // row_ptr[]
        + (long long)ncols * sizeof(float)     // x[]
        + (long long)nrows * sizeof(float);    // y[]

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    if (d_buffer)
        CUDA_CHECK(cudaFree(d_buffer));
    CUSPARSE_CHECK(cusparseDestroySpMat(matA));
    CUSPARSE_CHECK(cusparseDestroyDnVec(vecX));
    CUSPARSE_CHECK(cusparseDestroyDnVec(vecY));
    CUSPARSE_CHECK(cusparseDestroy(handle));
    CUDA_CHECK(cudaGetLastError());

    return KernelResults("cuSPARSE", avg_ms, min_ms, max_ms, variance_ms, bytes);
}

#include "spMVflat.hpp"
// ============================================================
// FLAT METHOD  (Chu et al., 2024)
// ============================================================

// NB: this code is a direct translation of the pseudocode in the Chu et al. paper.

/**
 * Preprocessing kernel.
 *
 * Fills bp[0..WGS-1] where bp[i] = row-id of the first NNZ
 * processed by workgroup (block) i.
 *
 * Each GPU thread handles one row i of the matrix:
 *  - cur_bp  = row_ptr[i] / STRIDE
 *  - next_bp = row_ptr[i+1] / STRIDE
 * If they differ, workgroups cur_bp+1 .. next_bp all start inside
 * row i, so bp[cur_bp+1..next_bp] = i.
 * Special case: if row_ptr[i+1] is exactly on a STRIDE boundary,
 * workgroup next_bp actually starts at row i+1 (add 1).
 */
__global__ void flat_preprocess(int m,
                                const int *__restrict__ row_ptr,
                                int *__restrict__ bp,
                                int STRIDE, // = N = R * THREADS -> number of NNZs per workgroup
                                int WGS)
{
    int total = gridDim.x * blockDim.x;
    int g_tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (g_tid == 0)
        bp[0] = 0;

    for (int i = g_tid; i < m; i += total)
    {
        int cur_bp = row_ptr[i] / STRIDE;
        int next_bp = row_ptr[i + 1] / STRIDE;

        if (cur_bp != next_bp)
        {
            for (int j = cur_bp + 1; j <= next_bp && j < WGS; j++)
                bp[j] = i;

            // Special case: row_ptr[i+1] exactly on boundary -> row i+1 starts
            if ((row_ptr[i + 1] % STRIDE == 0) && (next_bp < WGS))
                bp[next_bp] = bp[next_bp] + 1;
        }
    }
}

/**
 * Core Flat SpMV kernel.
 *
 * Each block processes exactly N = R * THREADS non-zeros (NNZ-splitting).
 *
 * Phase 1 - Load & multiply (coalesced):
 *   Each thread loads R elements into shared memory:
 *     shared_mem[thread + i*THREADS] = values[k] * x[col_idx[k]]
 *
 * Phase 2 - Reduction:
 *   Using bp[] we know which rows belong to this block.
 *   Each thread is assigned one (or more) rows and sums the
 *   corresponding entries from shared memory, then writes via atomicAdd.
 */
template <int R>
__global__ void spmv_flat(int m,
                          int nnz,
                          const int *__restrict__ row_ptr,
                          const int *__restrict__ col_idx,
                          const float *__restrict__ values,
                          const float *__restrict__ x,
                          float *__restrict__ y,
                          const int *__restrict__ bp,
                          int WGS)
{
    extern __shared__ float shared_mem[]; // size = N = R * THREADS

    const int THREADS = blockDim.x;
    const int N = R * THREADS;
    const int tid = threadIdx.x; // thread ID within block
    const int wg_id = blockIdx.x;
    const int wg_nnz_start = wg_id * N; // first global NNZ index for this block

    // ----------------------------------------------------------
    // Load R elements per thread into shared memory
    // ----------------------------------------------------------
    __syncthreads();

#pragma unroll
    for (int i = 0; i < R; i++)
    {
        int shared_idx = tid + i * THREADS;
        int global_idx = wg_nnz_start + shared_idx;

        // Clamp to avoid OOB; zero out padding beyond nnz
        int idx = min(global_idx, nnz - 1);
        float temp = values[idx] * x[col_idx[idx]];
        shared_mem[shared_idx] = (global_idx < nnz) ? temp : 0.0f;
    }

    __syncthreads();

    // ----------------------------------------------------------
    // Determine row range to reduce for this block
    // ----------------------------------------------------------
    int bp_idx = wg_id;

    int reduce_row_start = min(bp[bp_idx], m);
    int reduce_row_end = (bp_idx + 1 < WGS) ? min(bp[bp_idx + 1], m) : m;

    // Last workgroup guard
    if (reduce_row_end == 0)
        reduce_row_end = m;

    // Extend by 1 if next WG starts exactly on boundary OR single-row case
    // NB bug fix: in the paper it was: (row_ptr[reduce_row_end] % N == 0...
    if (reduce_row_end < m &&
        (row_ptr[reduce_row_end] % N != 0 || reduce_row_start == reduce_row_end))
        reduce_row_end = min(reduce_row_end + 1, m);

    // ----------------------------------------------------------
    // Each thread reduces its assigned rows and atomicAdds
    // ----------------------------------------------------------
    int reduce_row_id = reduce_row_start + tid;
    int bp_nnz_id = bp_idx * N; // global NNZ offset for this workgroup

    while (reduce_row_id < reduce_row_end)
    {
        float sum = 0.0f;

        int reduce_id_start = max(0, row_ptr[reduce_row_id] - bp_nnz_id);
        int reduce_id_end = min(N, row_ptr[reduce_row_id + 1] - bp_nnz_id);

        for (int i = reduce_id_start; i < reduce_id_end; i++)
            sum += shared_mem[i];

        atomicAdd(&y[reduce_row_id], sum); // multiple threads may update same row -> atomicAdd
        reduce_row_id += THREADS;
    }
}

KernelResults spmv_flat_launcher(const CSRDevice &d_csr,
                                 const float *d_x,
                                 float *d_y)
{
    int nrows = d_csr.nrows;
    int ncols = d_csr.ncols;
    int nnz = d_csr.nnz;

    const int THREADS = 256;
    const int R = 4;
    const int N = R * THREADS; // NNZs per workgroup

    int WGS = (nnz + N - 1) / N;

    // break-point array
    int *d_bp;
    CUDA_CHECK(cudaMalloc(&d_bp, (WGS + 1) * sizeof(int)));
    CUDA_CHECK(cudaMemset(d_bp, 0, (WGS + 1) * sizeof(int)));

    // ----------------------------------------------------------
    // Preprocessing (GPU-side, lightweight)
    // ----------------------------------------------------------
    cudaEvent_t p_start, p_stop;
    CUDA_CHECK(cudaEventCreate(&p_start));
    CUDA_CHECK(cudaEventCreate(&p_stop));

    int prep_threads = 256;
    int prep_blocks = (nrows + prep_threads - 1) / prep_threads;

    CUDA_CHECK(cudaEventRecord(p_start));
    flat_preprocess<<<prep_blocks, prep_threads>>>(
        nrows, d_csr.row_ptr, d_bp, N, WGS);
    CUDA_CHECK(cudaEventRecord(p_stop));
    CUDA_CHECK(cudaEventSynchronize(p_stop));

    float preprocess_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&preprocess_ms, p_start, p_stop));

    CUDA_CHECK(cudaEventDestroy(p_start));
    CUDA_CHECK(cudaEventDestroy(p_stop));

    // ----------------------------------------------------------
    // Benchmarking
    // ----------------------------------------------------------
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    int shmem_size = N * sizeof(float);

    // Warm-up
    for (int i = 0; i < WARMUP_ITERATIONS; i++)
    {
        CUDA_CHECK(cudaMemset(d_y, 0, nrows * sizeof(float)));
        spmv_flat<R><<<WGS, THREADS, shmem_size>>>(
            nrows, nnz,
            d_csr.row_ptr, d_csr.col_idx, d_csr.values,
            d_x, d_y, d_bp, WGS);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // Kernel benchmark
    float total_kernel_ms = 0.0f, min_ms = 1e30f, max_ms = 0.0f;
    float total_ms2 = 0.0f;

    for (int i = 0; i < BENCHMARK_ITERATIONS; i++)
    {
        CUDA_CHECK(cudaMemset(d_y, 0, nrows * sizeof(float)));

        CUDA_CHECK(cudaEventRecord(start));
        spmv_flat<R><<<WGS, THREADS, shmem_size>>>(
            nrows, nnz,
            d_csr.row_ptr, d_csr.col_idx, d_csr.values,
            d_x, d_y, d_bp, WGS);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        total_kernel_ms += ms;
        total_ms2 += ms * ms;
        min_ms = fminf(min_ms, ms);
        max_ms = fmaxf(max_ms, ms);
    }

    float avg_kernel_ms = total_kernel_ms / BENCHMARK_ITERATIONS;
    float variance_kernel_ms = total_ms2 / BENCHMARK_ITERATIONS - avg_kernel_ms * avg_kernel_ms;
    float total_end_to_end_ms = preprocess_ms + avg_kernel_ms;

    printf("GPU SpMV (FLAT):\n");
    printf("  WGS (blocks):            %d\n", WGS);
    printf("  N (NNZ per WG):          %d (R=%d, THREADS=%d)\n", N, R, THREADS);

    printf("\n  [Preprocessing]\n");
    printf("  time:                    %f ms\n", preprocess_ms);

    printf("\n  [Kernel]\n");
    printf("  avg:                     %f ms\n", avg_kernel_ms);
    printf("  min:                     %f ms\n", min_ms);
    printf("  max:                     %f ms\n\n", max_ms);
    printf("  variance:                %f ms*ms\n", variance_kernel_ms);

    printf("\n  [End-to-end]\n");
    printf("  preprocess + avg kernel: %f ms\n", total_end_to_end_ms);
    printf("  preprocess weight on total time: %f %%\n", preprocess_ms / total_end_to_end_ms * 100.0f);

    // Preprocessing bytes
    long long bytes_prep =
        (long long)(nrows + 1) * sizeof(int) // row_ptr[]: read to fill bp[]
        + (long long)WGS * sizeof(int);      // bp[]: write of the break-point array

    // Bytes of the actual SpMV kernel
    long long bytes_kernel =
        (long long)nnz * sizeof(float)         // values[]: one float per NNZ
        + (long long)nnz * sizeof(int)         // col_idx[]: one int per NNZ
        + (long long)(nrows + 1) * sizeof(int) // row_ptr[]: reread for the reduction phase
        + (long long)ncols * sizeof(float)     // x[]: upper bound = all columns
        + (long long)nrows * sizeof(float)     // y[]: write (atomicAdd) per row
        + (long long)WGS * sizeof(int);        // bp[]: read of the break-point array

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_bp));
    CUDA_CHECK(cudaGetLastError());

    KernelResults flat_res_e2e("FLAT (end-to-end)", total_end_to_end_ms, min_ms + preprocess_ms, max_ms + preprocess_ms, variance_kernel_ms, bytes_prep + bytes_kernel);

    return flat_res_e2e;
}

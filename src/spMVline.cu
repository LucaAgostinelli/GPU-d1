#include "spMVline.hpp"

// ============================================================
// LINE-ENHANCE METHOD  (Chu et al., 2024)
// ============================================================
//
// NB: this code is a direct translation of the pseudocode in the Chu et al. paper.
//
// Template parameters
//   R : NNZs loaded per thread per round  (paper uses R = 2)
//   V : threads per reduction vector (paper: power of 2, 1..THREADS/N)
//
// Each workgroup (block) processes exactly rows_per_wg complete rows
// (= N in the paper).  Because rows are never split across workgroups,
// no atomicAdd on the output vector y[] is needed.
//
// Execution flow per block
// ------------------------------------------------------------------
//  Initialisation
//    row_begin = wg_id * rows_per_wg
//    row_end = min(row_begin + rows_per_wg, nrows)
//    nnz_start = row_ptr[row_begin]
//    nnz_end = row_ptr[row_end]
//    rounds = ceil((nnz_end - nnz_start) / (R * THREADS))
//
//  For r = 0 .. rounds-1
//    Phase 1 - Multiplication (load --> shared_mem)
//      Each thread loads up to R elements with stride THREADS,
//      multiplies values[k] * x[col_idx[k]], stores in shared_mem.
//
//    syncthreads()
//
//    Phase 2 - VecReduce
//      Threads are divided into THREADS/V vectors of V threads each.
//      Vector vec_i is responsible for row row_begin + vec_i.
//      Each thread in the vector sums the entries of its row that
//      fall inside the current round's shared_memory slice,
//      stepping with stride V.
//      Then a warp-shuffle prefix sum collapses V partial sums to
//      thread 0 of the vector, which accumulates into a local register.
//
//    syncthreads()   (shared_mem reuse next round)
//
//  Phase 3 - Store results
//    If V > 1, prefix-sum is already done inside VecReduce.
//    Thread 0 of each vector writes sum to y[reduce_row_id].
// ============================================================

// ------------------------------------------------------------------
// Warp-level inclusive prefix sum (used when V > 1)
// ------------------------------------------------------------------
__inline__ __device__ float warp_prefix_sum(float val, int lane, int width)
{
    for (int offset = 1; offset < width; offset <<= 1)
    {
        float n = __shfl_up_sync(0xffffffff, val, offset, width);
        if (lane >= offset)
            val += n;
    }
    return val;
}

// ------------------------------------------------------------------
// Core Line-enhance kernel
//
// Template params:
//   R  - rounds unroll factor (NNZ per thread per round)
//   V  - threads per reduction vector (power of 2, >= 1)
// ------------------------------------------------------------------
template <int R, int V>
__global__ void spmv_line_enhance(int nrows,
                                  int rows_per_wg,
                                  const int *__restrict__ row_ptr,
                                  const int *__restrict__ col_idx,
                                  const float *__restrict__ values,
                                  const float *__restrict__ x,
                                  float *__restrict__ y)
{
    extern __shared__ float shared_mem[]; // size = R * THREADS floats

    const int THREADS = blockDim.x;
    const int N_SHARED = R * THREADS; // NNZs covered per round

    const int tid = threadIdx.x;
    const int wg_id = blockIdx.x;

    // ----- Task partitioning -----
    const int row_begin = wg_id * rows_per_wg;
    const int row_end = min(row_begin + rows_per_wg, nrows);

    if (row_begin >= nrows)
        return;

    const int nnz_start = row_ptr[row_begin];
    const int nnz_end = row_ptr[row_end];
    const int local_nnz = nnz_end - nnz_start;

    // ----- Rounds calculation -----
    const int rounds = (local_nnz + N_SHARED - 1) / N_SHARED;

    // Each thread holds its private accumulator for V rows
    // (when V == 1 each thread owns 1 row; for V > 1 a vector owns 1 row)
    const int vec_i = tid / V;      // which vector (= which row offset) this thread belongs to
    const int tid_in_vec = tid % V; // lane within the vector

    // Running sum for this thread's vector
    float sum = 0.0f;

    // ----- Round loop -----
    for (int r = 0; r < rounds; ++r)
    {
        // ----- Phase 1: load & multiply into shared_mem -----
        const int round_inx_start = nnz_start + r * N_SHARED;
        const int round_inx_end = min(round_inx_start + N_SHARED, nnz_end);

        // Each thread loads R elements with stride THREADS
        int i = round_inx_start + tid;
#pragma unroll
        for (int k = 0; k < R; ++k)
        {
            int shared_slot = tid + k * THREADS; // position in shared_mem
            if (i < round_inx_end)
                shared_mem[shared_slot] = values[i] * x[col_idx[i]];
            else
                shared_mem[shared_slot] = 0.0f;
            i += THREADS;
        }

        __syncthreads();

        // ----- Phase 2: VecReduce -----
        int reduce_row_id = row_begin + vec_i;
        float local_sum = 0.0f;

        if (reduce_row_id < row_end)
        {
            // row range in global NNZ space
            int Ibegin = row_ptr[reduce_row_id];
            int Iend = row_ptr[reduce_row_id + 1];

            // clip to this round's shared_mem window
            int Rbegin = max(Ibegin, round_inx_start);
            int Rend = min(Iend, round_inx_end);

            // shared_mem offset for this round's window: always = r * N_SHARED
            int sm_base = round_inx_start - nnz_start;

            // stride by V inside the vector (Algorithm 5, line 7)
            for (int j = Rbegin + tid_in_vec; j < Rend; j += V)
                local_sum += shared_mem[j - nnz_start - sm_base];
        }

        // --- warp_prefix_sum is called UNCONDITIONALLY for all threads ---
        // Inactive threads (reduce_row_id >= row_end) contribute local_sum=0,
        // which does not corrupt active threads since __shfl width=V keeps
        // each V-group isolated. The result for inactive threads is discarded.
        if (V > 1)
        {
            // inclusive prefix sum; only lane V-1 of each V-group holds the total
            local_sum = warp_prefix_sum(local_sum, tid_in_vec, V);
            if (tid_in_vec == V - 1 && reduce_row_id < row_end)
                sum += local_sum;
        }
        else
        {
            // V == 1: no shfl needed, each thread owns its own row
            if (reduce_row_id < row_end)
                sum += local_sum;
        }

        __syncthreads(); // shared_mem can be reused next round
    }

    // ----- Phase 3: store results -----
    // Thread 0 of each vector (or tid_in_vec == V-1 when V > 1) writes y.
    // For V == 1 every thread is its own vector leader.
    bool is_leader = (V == 1) ? true : (tid_in_vec == V - 1);

    int reduce_row_id = row_begin + vec_i;
    if (is_leader && reduce_row_id < row_end)
        y[reduce_row_id] = sum;
}

KernelResults spmv_line_launcher(const CSRDevice &d_csr,
                                 const float *d_x,
                                 float *d_y)
{
    int nrows = d_csr.nrows;
    int ncols = d_csr.ncols;
    int nnz = d_csr.nnz;

    const int THREADS = 512;
    const int R = 2;
    const int N_SHARED = R * THREADS; // NNZs per round per block

    // Choose V (threads per reduction vector) based on avg NNZ/row:
    float avg_nnz = (nrows > 0) ? (float)nnz / nrows : 1.0f;

    int V;
    if (avg_nnz >= 24.0f)
        V = 4;
    else if (avg_nnz >= 8.0f)
        V = 2;
    else
        V = 1;

    int rows_per_wg = THREADS / V;

    // Number of workgroups (blocks)
    int WGS = (nrows + rows_per_wg - 1) / rows_per_wg;

    int shmem_size = N_SHARED * sizeof(float);

    printf("GPU SpMV (LINE-ENHANCE):\n");
    printf("  avg NNZ/row: %.2f --> V=%d, rows_per_wg=%d\n",
           avg_nnz, V, rows_per_wg);
    printf("  THREADS=%d, R=%d, N_SHARED=%d, WGS=%d\n",
           THREADS, R, N_SHARED, WGS);

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Warm-up
    for (int i = 0; i < WARMUP_ITERATIONS; ++i)
    {
        CUDA_CHECK(cudaMemset(d_y, 0, nrows * sizeof(float)));
        if (V == 1)
            spmv_line_enhance<R, 1><<<WGS, THREADS, shmem_size>>>(nrows, rows_per_wg, d_csr.row_ptr, d_csr.col_idx, d_csr.values, d_x, d_y);
        else if (V == 2)
            spmv_line_enhance<R, 2><<<WGS, THREADS, shmem_size>>>(nrows, rows_per_wg, d_csr.row_ptr, d_csr.col_idx, d_csr.values, d_x, d_y);
        else
            spmv_line_enhance<R, 4><<<WGS, THREADS, shmem_size>>>(nrows, rows_per_wg, d_csr.row_ptr, d_csr.col_idx, d_csr.values, d_x, d_y);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // Benchmarking
    float total_ms = 0.0f, min_ms = 1e30f, max_ms = 0.0f;
    float total_ms2 = 0.0f;

    for (int i = 0; i < BENCHMARK_ITERATIONS; ++i)
    {
        CUDA_CHECK(cudaMemset(d_y, 0, nrows * sizeof(float)));
        CUDA_CHECK(cudaEventRecord(start));

        if (V == 1)
            spmv_line_enhance<R, 1><<<WGS, THREADS, shmem_size>>>(nrows, rows_per_wg, d_csr.row_ptr, d_csr.col_idx, d_csr.values, d_x, d_y);
        else if (V == 2)
            spmv_line_enhance<R, 2><<<WGS, THREADS, shmem_size>>>(nrows, rows_per_wg, d_csr.row_ptr, d_csr.col_idx, d_csr.values, d_x, d_y);
        else
            spmv_line_enhance<R, 4><<<WGS, THREADS, shmem_size>>>(nrows, rows_per_wg, d_csr.row_ptr, d_csr.col_idx, d_csr.values, d_x, d_y);

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
    long long bytes =
        (long long)nnz * sizeof(float) // values[]: one float per NNZ
        + (long long)nnz * sizeof(int) // col_idx[]: one int per NNZ
        // row_ptr read twice per row: start and end of each row in every round
        + (long long)(nrows + 1) * sizeof(int) // row_ptr[]: accessed by VecReduce for each row
        + (long long)ncols * sizeof(float)     // x[]: all columns
        + (long long)nrows * sizeof(float);    // y[]: direct write (no atomic)
    // NB: shared memory is on-chip --> not counted in bandwidth calculation

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaGetLastError());

    return KernelResults("LINE-ENHANCE", avg_ms, min_ms, max_ms, variance_ms, bytes);
}
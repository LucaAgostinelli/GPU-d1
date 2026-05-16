#include "spMVsv-adaptive.hpp"
#include <vector>

// ==============================================================================
// Scalar/Vectore ADAPTIVE SpMV
// ==============================================================================
//
// Preprocessing (CPU, O(nrows)):
//   Scan row_ptr once and split row indices into two arrays:
//     short_rows[] : rows with nnz <  SHORT_THRESHOLD  (scalar kernel)
//     long_rows[]  : rows with nnz >= SHORT_THRESHOLD  (vector kernel)
//
// Kernel A - spmv_csr_scalar (short rows):
//   1 thread per row, simple sequential loop over the row's NNZs.
//   gridDim  = ceil(n_short / BLOCK_SCALAR)
//   blockDim = BLOCK_SCALAR  (e.g. 256)
//
// Kernel B - spmv_csr_vector (long rows):
//   1 warp (32 threads) per row, strided access + warp-shuffle reduction.
//   gridDim  = ceil(n_long * 32 / BLOCK_VECTOR)
//   blockDim = BLOCK_VECTOR  (e.g. 256 --> 8 warps per block)
//
// The two kernels are launched on the same (default) stream sequentially,
// so they never race on y[].
//
// SHORT_THRESHOLD = 32
//   Chosen to match the warp size: a row with < 32 NNZs cannot keep a full
//   warp busy for even a single iteration, so the scalar path is cheaper.
//
// ==============================================================================

static constexpr int SHORT_THRESHOLD = 32;
static constexpr int BLOCK_SCALAR = 256;
static constexpr int BLOCK_VECTOR = 256;

// ------------------------------------------------------------------
// Kernel A: scalar - one thread per short row
// ------------------------------------------------------------------
__global__ void spmv_csr_scalar(const int *__restrict__ short_rows, // [n_short]
                                int n_short,
                                const int *__restrict__ row_ptr,
                                const int *__restrict__ col_idx,
                                const float *__restrict__ values,
                                const float *__restrict__ x,
                                float *__restrict__ y)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_short)
        return;

    int row = short_rows[tid];
    int row_start = row_ptr[row];
    int row_end = row_ptr[row + 1];

    float sum = 0.0f;
    for (int i = row_start; i < row_end; ++i)
        sum += values[i] * x[col_idx[i]];

    y[row] = sum;
}

// ------------------------------------------------------------------
// Kernel B: vector - one warp per long row
// ------------------------------------------------------------------
__device__ __forceinline__ float warp_reduce_sum_adaptive(float val)
{
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

__global__ void spmv_csr_vector(const int *__restrict__ long_rows, // [n_long]
                                int n_long,
                                const int *__restrict__ row_ptr,
                                const int *__restrict__ col_idx,
                                const float *__restrict__ values,
                                const float *__restrict__ x,
                                float *__restrict__ y)
{
    // Each warp handles one entry in long_rows[]
    int global_thread = blockIdx.x * blockDim.x + threadIdx.x;
    int warp_id = global_thread / 32;
    int lane = threadIdx.x & 31;

    if (warp_id >= n_long)
        return;

    int row = long_rows[warp_id];
    int row_start = row_ptr[row];
    int row_end = row_ptr[row + 1];

    float sum = 0.0f;
    for (int i = row_start + lane; i < row_end; i += 32)
        sum += values[i] * x[col_idx[i]];

    sum = warp_reduce_sum_adaptive(sum);

    if (lane == 0)
        y[row] = sum;
}

KernelResults spmv_sv_adaptive_launcher(const CSRDevice &d_csr,
                                        const CSRHost &h_csr,
                                        const float *d_x,
                                        float *d_y)
{
    int nrows = h_csr.nrows;
    int ncols = h_csr.ncols;
    int nnz = h_csr.nnz;

    // Preprocessing: split rows into short / long  (O(nrows))
    std::vector<int> h_short_rows, h_long_rows;
    h_short_rows.reserve(nrows);
    h_long_rows.reserve(nrows);

    for (int i = 0; i < nrows; ++i)
    {
        int nnz_row = h_csr.row_ptr[i + 1] - h_csr.row_ptr[i];
        if (nnz_row < SHORT_THRESHOLD)
            h_short_rows.push_back(i);
        else
            h_long_rows.push_back(i);
    }

    int n_short = (int)h_short_rows.size();
    int n_long = (int)h_long_rows.size();

    printf("GPU SpMV (S-V-ADAPTIVE - two-kernel):\n");
    printf("  SHORT_THRESHOLD = %d\n", SHORT_THRESHOLD);
    printf("  short rows (scalar kernel): %d  (%.1f%%)\n",
           n_short, 100.0f * n_short / nrows);
    printf("  long  rows (vector   kernel): %d  (%.1f%%)\n",
           n_long, 100.0f * n_long / nrows);

    // Upload index arrays to device
    int *d_short_rows = nullptr, *d_long_rows = nullptr;
    if (n_short > 0)
    {
        CUDA_CHECK(cudaMalloc(&d_short_rows, n_short * sizeof(int)));
        CUDA_CHECK(cudaMemcpy(d_short_rows, h_short_rows.data(),
                              n_short * sizeof(int), cudaMemcpyHostToDevice));
    }
    if (n_long > 0)
    {
        CUDA_CHECK(cudaMalloc(&d_long_rows, n_long * sizeof(int)));
        CUDA_CHECK(cudaMemcpy(d_long_rows, h_long_rows.data(),
                              n_long * sizeof(int), cudaMemcpyHostToDevice));
    }

    // Grid dimensions
    int grid_scalar = (n_short + BLOCK_SCALAR - 1) / BLOCK_SCALAR;
    // Each long row needs a full warp (32 threads)
    int total_warp_threads = n_long * 32;
    int grid_vector = (total_warp_threads + BLOCK_VECTOR - 1) / BLOCK_VECTOR;

    auto launch_both = [&]()
    {
        if (n_short > 0 && grid_scalar > 0)
            spmv_csr_scalar<<<grid_scalar, BLOCK_SCALAR>>>(
                d_short_rows, n_short,
                d_csr.row_ptr, d_csr.col_idx, d_csr.values,
                d_x, d_y);

        if (n_long > 0 && grid_vector > 0)
            spmv_csr_vector<<<grid_vector, BLOCK_VECTOR>>>(
                d_long_rows, n_long,
                d_csr.row_ptr, d_csr.col_idx, d_csr.values,
                d_x, d_y);
    };

    // Warm-up
    for (int i = 0; i < WARMUP_ITERATIONS; ++i)
    {
        CUDA_CHECK(cudaMemset(d_y, 0, nrows * sizeof(float)));
        launch_both();
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // Benchmarking
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    float total_ms = 0.0f, min_ms = 1e30f, max_ms = 0.0f;
    float total_ms2 = 0.0f;

    for (int i = 0; i < BENCHMARK_ITERATIONS; ++i)
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

    float avg_ms = total_ms / BENCHMARK_ITERATIONS;
    float variance_ms = total_ms2 / BENCHMARK_ITERATIONS - avg_ms * avg_ms;
    long long bytes = (long long)nnz * sizeof(float)         // values[]: one float per NNZ
                      + (long long)nnz * sizeof(int)         // col_idx[]: one int per NNZ
                      + (long long)(nrows + 1) * sizeof(int) // row_ptr[]: one entry per row + 1
                      + (long long)ncols * sizeof(float)     // x[]: upper bound = all columns
                      + (long long)nrows * sizeof(float)     // y[]: one float written per row
                      // auxiliary arrays copied to the device during preprocessing
                      + (long long)n_short * sizeof(int) // short_rows[]: indices of scalar rows
                      + (long long)n_long * sizeof(int); // long_rows[]: indices of warp rows

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    if (d_short_rows)
        CUDA_CHECK(cudaFree(d_short_rows));
    if (d_long_rows)
        CUDA_CHECK(cudaFree(d_long_rows));
    CUDA_CHECK(cudaGetLastError());

    return KernelResults("S-V-ADAPTIVE", avg_ms, min_ms, max_ms, variance_ms, bytes);
}
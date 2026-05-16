#include "spMVacc.hpp"
#include "spMVflat.hpp"
#include "spMVline.hpp"
#include "optional"

KernelResults spmv_acc_launcher(const CSRDevice &d_csr,
                                const CSRHost &h_csr,
                                const float *d_x,
                                float *d_y,
                                bool test_mode)
{
    const int nrows = d_csr.nrows;
    const int nnz = d_csr.nnz;
    const float avg_nnz = (nrows > 0) ? (float)nnz / nrows : 1.0f;

    // ----- Heuristic: decide which kernel to use -----
    bool use_line = true;

    int max_row_nnz = 0;
    for (int i = 0; i < nrows; i++)
        max_row_nnz = std::max(max_row_nnz, h_csr.row_ptr[i + 1] - h_csr.row_ptr[i]);

    const float imbalance = (float)max_row_nnz / avg_nnz;

    if (nnz > 1'000'000)
    {
        if (imbalance > 32.0f || avg_nnz > 8.0f)
        {
            printf("  [ACC -> FLAT: imbalance=%.1f > 32 || avg_nnz=%.1f > 8]\n", imbalance, avg_nnz);
            use_line = false;
        }
    }

    if (avg_nnz > 128.0f)
    {
        printf("  [ACC -> FLAT: avg_nnz=%.1f > 128]\n", avg_nnz);
        use_line = false;
    }

    // ----- Allocate output buffers -----
    // In test_mode both kernels and compare; otherwise only the chosen one.
    float *d_y_line = nullptr;
    float *d_y_flat = nullptr;

    if (test_mode)
    {
        CUDA_CHECK(cudaMalloc(&d_y_line, nrows * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_y_flat, nrows * sizeof(float)));
        CUDA_CHECK(cudaMemset(d_y_line, 0, nrows * sizeof(float)));
        CUDA_CHECK(cudaMemset(d_y_flat, 0, nrows * sizeof(float)));
    }

    // ----- Run kernel(s) -----
    KernelResults res_line("LINE", 0, 0, 0, 0, 0);
    KernelResults res_flat("FLAT", 0, 0, 0, 0, 0);

    if (test_mode)
    {
        printf("  [ACC TEST] running LINE...\n");
        res_line = spmv_line_launcher(d_csr, d_x, d_y_line);

        printf("  [ACC TEST] running FLAT...\n");
        res_flat = spmv_flat_launcher(d_csr, d_x, d_y_flat);
    }
    else if (use_line)
    {
        printf("  [ACC -> LINE]\n");
        res_line = spmv_line_launcher(d_csr, d_x, d_y);
    }
    else
    {
        printf("  [ACC -> FLAT]\n");
        res_flat = spmv_flat_launcher(d_csr, d_x, d_y);
    }

    // ----- test_mode: correctness check + heuristic validation -----
    const KernelResults &chosen =
        (use_line ? res_line : res_flat);

    if (test_mode)
    {
        // Copy results to host for comparison
        std::vector<float> h_line(nrows), h_flat(nrows);
        CUDA_CHECK(cudaMemcpy(h_line.data(), d_y_line,
                              nrows * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_flat.data(), d_y_flat,
                              nrows * sizeof(float), cudaMemcpyDeviceToHost));

        // Max absolute difference between the two outputs
        float max_diff = 0.0f;
        for (int i = 0; i < nrows; i++)
            max_diff = std::max(max_diff, fabsf(h_line[i] - h_flat[i]));

        const bool line_faster = (res_line.avg_ms < res_flat.avg_ms);
        const bool heuristic_correct =
            (use_line && line_faster) || (!use_line && !line_faster);

        printf("\n[ACC TEST RESULT]\n");
        printf("  avg_nnz=%.2f  imbalance=%.2f\n", avg_nnz, imbalance);
        printf("  LINE  time: %.4f ms\n", res_line.avg_ms);
        printf("  FLAT  time: %.4f ms\n", res_flat.avg_ms);
        printf("  max |line - flat|: %e\n", max_diff);
        printf("  heuristic choice:  %s\n", use_line ? "LINE" : "FLAT");
        printf("  best choice:       %s\n", line_faster ? "LINE" : "FLAT");
        printf("  heuristic:         %s\n", heuristic_correct ? "CORRECT" : "WRONG");

        // Copy the winner's output into d_y
        float *d_y_winner = use_line ? d_y_line : d_y_flat;
        CUDA_CHECK(cudaMemcpy(d_y, d_y_winner,
                              nrows * sizeof(float), cudaMemcpyDeviceToDevice));

        CUDA_CHECK(cudaFree(d_y_line));
        CUDA_CHECK(cudaFree(d_y_flat));
    }

    return chosen;
}
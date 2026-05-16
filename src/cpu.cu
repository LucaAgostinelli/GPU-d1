#include "cpu.hpp"

void spmv_cpu(int nrows,
              const CSRHost &csr,
              const std::vector<float> &x,
              std::vector<float> &y)
{
    for (int row = 0; row < nrows; row++)
    {
        float sum = 0.0f;

        for (int nnz_idx = csr.row_ptr[row]; nnz_idx < csr.row_ptr[row + 1]; nnz_idx++)
        {
            sum += csr.values[nnz_idx] * x[csr.col_idx[nnz_idx]];
        }

        y[row] = sum;
    }
}

void benchmark_cpu_spmv(int nrows,
                        const CSRHost &csr,
                        const std::vector<float> &h_x,
                        std::vector<float> &h_y_cpu)
{
    int cpu_runs = 10;

    double total = 0.0;
    double min_t = 1e30;
    double max_t = 0.0;

    TIMER_DEF;

    for (int i = 0; i < cpu_runs; i++)
    {
        TIMER_START;
        spmv_cpu(nrows, csr, h_x, h_y_cpu);
        TIMER_STOP;

        double t = TIMER_ELAPSED;

        total += t;
        min_t = std::min(min_t, t);
        max_t = std::max(max_t, t);
    }

    // printf("CPU SpMV:\n");
    // printf("  avg: %f s\n", total / cpu_runs);
    // printf("  min: %f s\n", min_t);
    // printf("  max: %f s\n", max_t);
}

bool almost_equal(float a, float b, float abs_tol, float rel_tol)
{
    float diff = fabsf(a - b);

    if (diff <= abs_tol)
        return true;

    return diff <= rel_tol * fmaxf(fabsf(a), fabsf(b));
}

void check_correctness(int nrows,
                       const std::vector<float> &h_y_cpu,
                       const std::vector<float> &h_y_gpu)
{
    int errors = 0;

    for (int i = 0; i < nrows; i++)
    {
        if (!almost_equal(h_y_cpu[i], h_y_gpu[i]))
        {
            if (errors < 10)
                printf("[ERROR CHECKING] Mismatch at index %d: CPU = %f, GPU = %f\n",
                       i, h_y_cpu[i], h_y_gpu[i]);
            errors++;
        }
    }

    if (errors > 0)
    {
        printf("[ERROR CHECKING] Total mismatches: %d\n", errors);
    } else {
        printf("[ERROR CHECKING] OK!\n");
    }
}

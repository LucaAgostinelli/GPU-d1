#include <stdio.h>
#include <stdlib.h>
#include <climits>
#include <cmath>

#include "matrix.hpp"
#include "utils.hpp"

// nvcc -O3 -std=c++17 src/matrix_stats.cu src/matrix.cu src/utils.cu -Iinclude -o bin/matrix_stats

int main(int argc, char **argv)
{
    TIMER_DEF;
    int nrows, ncols;
    bool symmetric;

    if (argc < 2)
    {
        fprintf(stderr, "Usage: %s <matrix_path>\n", argv[0]);
        return 1;
    }

    std::string path = argv[1];

    TIMER_START;
    auto coo = read_mtx(path, nrows, ncols, symmetric);
    CSRHost h_csr = coo_to_csr(coo, nrows, ncols);
    TIMER_STOP;

    float parse_time = TIMER_ELAPSED;

    int max_row_nnz = 0;
    int min_row_nnz = INT_MAX;
    long long nnz_sq_sum = 0;

    for (int i = 0; i < nrows; i++)
    {
        int row_nnz = h_csr.row_ptr[i + 1] - h_csr.row_ptr[i];

        if (row_nnz > max_row_nnz)
            max_row_nnz = row_nnz;

        if (row_nnz < min_row_nnz)
            min_row_nnz = row_nnz;

        nnz_sq_sum += (long long)row_nnz * row_nnz;
    }

    float avg_nnz = (float)h_csr.nnz / nrows;
    float variance = (float)nnz_sq_sum / nrows - avg_nnz * avg_nnz;
    float std_nnz = sqrtf(variance < 0.0f ? 0.0f : variance);
    float imbalance = (avg_nnz > 0.0f) ? (float)max_row_nnz / avg_nnz : 0.0f;

    printf("MATRIX,%s,%d,%d,%d,%d,%.6f,%.6f,%d,%d,%.6f,%.6f\n",
           path.c_str(),
           nrows,
           ncols,
           h_csr.nnz,
           (int)symmetric,
           parse_time,
           avg_nnz,
           min_row_nnz,
           max_row_nnz,
           std_nnz,
           imbalance);

    return 0;
}
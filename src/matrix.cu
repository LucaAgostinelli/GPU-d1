#include "matrix.hpp"
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <algorithm>
#include <string>
#include <cuda_runtime.h>

// ============================================================
// PARSER
// ============================================================

// NB: unsorted COO
std::vector<COO> read_mtx(const std::string &filename,
                          int &nrows, int &ncols,
                          bool &symmetric)
{
    std::ifstream file(filename);
    if (!file)
    {
        throw std::runtime_error("Cannot open file: " + filename);
    }

    std::string line;

    std::getline(file, line);
    symmetric = (line.find("symmetric") != std::string::npos);

    while (std::getline(file, line))
    {
        if (line[0] != '%')
            break;
    }

    std::stringstream ss(line);
    int nnz;
    ss >> nrows >> ncols >> nnz;

    std::vector<COO> coo;
    coo.reserve(nnz * (symmetric ? 2 : 1));

    int i, j;
    float val;

    while (file >> i >> j)
    {
        if (!(file >> val))
            val = 1.0f; // pattern matrix

        // 0-based
        i--;
        j--;

        coo.push_back({i, j, val});

        // symmetric matrix
        if (symmetric && i != j)
        {
            coo.push_back({j, i, val});
        }
    }

    return coo;
}

// ============================================================
// CSR
// ============================================================

CSRHost coo_to_csr(std::vector<COO> &coo, int nrows, int ncols)
{
    CSRHost csr;
    csr.nrows = nrows;
    csr.ncols = ncols;
    csr.nnz = coo.size();

    std::sort(coo.begin(), coo.end(), [](const COO &a, const COO &b)
              { return (a.row < b.row) || (a.row == b.row && a.col < b.col); });

    csr.row_ptr.resize(nrows + 1, 0);
    csr.col_idx.resize(csr.nnz);
    csr.values.resize(csr.nnz);

    // nnz per row
    for (const COO &e : coo)
    {
        csr.row_ptr[e.row + 1]++;
    }

    // Prefix sum
    for (int i = 0; i < nrows; i++)
    {
        csr.row_ptr[i + 1] += csr.row_ptr[i];
    }

    std::vector<int> offset = csr.row_ptr;

    for (const COO &e : coo)
    {
        int pos = offset[e.row]++;
        csr.col_idx[pos] = e.col;
        csr.values[pos] = e.val;
    }

    return csr;
}

CSRDevice csr_host_to_device(const CSRHost &h_csr)
{
    CSRDevice d_csr;
    d_csr.nrows = h_csr.nrows;
    d_csr.ncols = h_csr.ncols;
    d_csr.nnz = h_csr.nnz;

    cudaMalloc(&d_csr.row_ptr, (h_csr.nrows + 1) * sizeof(int));
    cudaMalloc(&d_csr.col_idx, h_csr.nnz * sizeof(int));
    cudaMalloc(&d_csr.values, h_csr.nnz * sizeof(float));

    cudaMemcpy(d_csr.row_ptr, h_csr.row_ptr.data(),
               (h_csr.nrows + 1) * sizeof(int), cudaMemcpyHostToDevice);

    cudaMemcpy(d_csr.col_idx, h_csr.col_idx.data(),
               h_csr.nnz * sizeof(int), cudaMemcpyHostToDevice);

    cudaMemcpy(d_csr.values, h_csr.values.data(),
               h_csr.nnz * sizeof(float), cudaMemcpyHostToDevice);

    return d_csr;
}

// ============================================================
// ELL
// ============================================================

// COLUMN-MAJOR
//   Element (row, c) is at index  c * nrows + row
//
// This guarantees that a warp reading the c-th column slot for
// 32 consecutive rows accesses 32 consecutive memory locations
// --> fully coalesced load on the GPU.
//
// Padding value for col_idx: 0 (a valid column index whose
// contribution is zeroed by values = 0.0f, so the dot product
// is unaffected regardless of x[0]).
//
ELLHost csr_to_ell(const CSRHost &h_csr, float &fill_ratio)
{
    const int nrows = h_csr.nrows;

    // ----- Find max NNZ per row -----
    int max_col = 0;
    for (int i = 0; i < nrows; i++)
    {
        int row_nnz = h_csr.row_ptr[i + 1] - h_csr.row_ptr[i];
        if (row_nnz > max_col)
            max_col = row_nnz;
    }

    // ----- Safety check: reject matrices where ELL would be impractical -----
    // ELL needs  max_col * nrows * 8 bytes (4 for col_idx + 4 for values).
    // I refuse to allocate more than MAX_ELL_BYTES on the host to avoid
    // std::bad_alloc or OOM-kill.  The caller (launcher) receives fill_ratio=-1
    // as a sentinel and should skip the kernel entirely
    static constexpr size_t MAX_ELL_BYTES = 4ULL * 1024 * 1024 * 1024; // 4 GB
    const size_t total = (size_t)max_col * nrows;
    const size_t bytes = total * (sizeof(int) + sizeof(float)); // 8 bytes/slot

    if (bytes > MAX_ELL_BYTES)
    {
        fill_ratio = -1.0f; // sentinel: ELL not feasible
        // Return an empty (but valid) struct so the caller can check fill_ratio
        ELLHost empty{};
        empty.nrows = nrows;
        empty.ncols = h_csr.ncols;
        empty.max_col = 0;
        return empty;
    }

    ELLHost ell;
    ell.nrows = nrows;
    ell.ncols = h_csr.ncols;
    ell.max_col = max_col;

    // Allocate and zero-initialise (padding = 0)
    ell.col_idx.assign(total, 0);
    ell.values.assign(total, 0.0f);

    // ----- Fill column-major -----
    // For row i, its c-th NNZ goes to slot  c * nrows + i
    for (int i = 0; i < nrows; i++)
    {
        int c = 0;
        for (int ptr = h_csr.row_ptr[i]; ptr < h_csr.row_ptr[i + 1]; ptr++, c++)
        {
            ell.col_idx[c * nrows + i] = h_csr.col_idx[ptr];
            ell.values[c * nrows + i] = h_csr.values[ptr];
        }
    }

    // fill_ratio = how many of the allocated slots are real NNZs
    fill_ratio = (max_col > 0)
                     ? (float)h_csr.nnz / (float)total
                     : 1.0f;

    return ell;
}

ELLDevice ell_host_to_device(const ELLHost &h_ell)
{
    ELLDevice d_ell;
    d_ell.nrows = h_ell.nrows;
    d_ell.ncols = h_ell.ncols;
    d_ell.max_col = h_ell.max_col;

    const size_t total = (size_t)h_ell.max_col * h_ell.nrows;

    cudaMalloc(&d_ell.col_idx, total * sizeof(int));
    cudaMalloc(&d_ell.values, total * sizeof(float));

    cudaMemcpy(d_ell.col_idx, h_ell.col_idx.data(),
               total * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ell.values, h_ell.values.data(),
               total * sizeof(float), cudaMemcpyHostToDevice);

    return d_ell;
}

void ell_device_free(ELLDevice &d_ell)
{
    if (d_ell.col_idx)
    {
        cudaFree(d_ell.col_idx);
        d_ell.col_idx = nullptr;
    }
    if (d_ell.values)
    {
        cudaFree(d_ell.values);
        d_ell.values = nullptr;
    }
}

// ============================================================
// HYB
// ============================================================

// ------------------------------------------------------------------
// Helper: choose K (ELL width) using Bell & Garland's criterion.
//
// Build a histogram of row lengths, then scan K = 1, 2, 3,...
// and keep adding a column to ELL as long as:
// rows_with_at_least_K_nnz  >=  max(4096, nrows / 3)
//
// ELL is modelled as ~3× faster than COO when fully occupied.
// A K-th column is "worth it" when at least 1/3 of the
// rows actually have K or more NNZs (otherwise too many padding
// zeros pollute the ELL pass).  The 4096 floor ensures that very
// small matrices still get a reasonable ELL portion.
// ------------------------------------------------------------------
static int choose_ell_cols(const CSRHost &h_csr)
{
    const int nrows = h_csr.nrows;

    // Find max row length
    int max_nnz_row = 0;
    for (int i = 0; i < nrows; i++)
    {
        int len = h_csr.row_ptr[i + 1] - h_csr.row_ptr[i];
        if (len > max_nnz_row)
            max_nnz_row = len;
    }

    if (max_nnz_row == 0)
        return 0;

    // Build histogram: hist[k] = number of rows with nnz == k
    std::vector<int> hist(max_nnz_row + 1, 0);
    for (int i = 0; i < nrows; i++)
    {
        int len = h_csr.row_ptr[i + 1] - h_csr.row_ptr[i];
        hist[len]++;
    }

    // Suffix sum: rows_ge[k] = number of rows with nnz >= k
    // I compute it incrementally: start from nrows (all rows have >= 0)
    // and subtract hist[k] at each step.
    int threshold = std::max(4096, nrows / 3);
    int rows_ge = nrows; // rows with nnz >= 0 (all)
    int K = 0;

    for (int k = 1; k <= max_nnz_row; k++)
    {
        rows_ge -= hist[k - 1]; // now rows_ge = count of rows with nnz >= k
        if (rows_ge < threshold)
            break;
        K = k;
    }

    return K;
}

HYBHost csr_to_hyb(const CSRHost &h_csr, int &ell_cols_out)
{
    const int nrows = h_csr.nrows;

    int K = choose_ell_cols(h_csr);
    ell_cols_out = K;

    HYBHost h_hyb;
    h_hyb.nrows = nrows;
    h_hyb.ncols = h_csr.ncols;
    h_hyb.ell_cols = K;

    // ----- Allocate ELL arrays (column-major, zero-padded) -----
    const size_t ell_total = (size_t)K * nrows;
    h_hyb.ell_col_idx.assign(ell_total, 0);
    h_hyb.ell_values.assign(ell_total, 0.0f);

    // ----- Collect COO overflow -----
    h_hyb.coo_nnz = 0;
    h_hyb.coo_row.clear();
    h_hyb.coo_col.clear();
    h_hyb.coo_val.clear();

    for (int i = 0; i < nrows; i++)
    {
        int ptr = h_csr.row_ptr[i];
        int row_end = h_csr.row_ptr[i + 1];
        int c = 0; // slot counter within the row

        for (; ptr < row_end; ptr++, c++)
        {
            if (c < K)
            {
                // ELL slot  c * nrows + i  (column-major)
                h_hyb.ell_col_idx[c * nrows + i] = h_csr.col_idx[ptr];
                h_hyb.ell_values[c * nrows + i] = h_csr.values[ptr];
            }
            else
            {
                // COO overflow
                h_hyb.coo_row.push_back(i);
                h_hyb.coo_col.push_back(h_csr.col_idx[ptr]);
                h_hyb.coo_val.push_back(h_csr.values[ptr]);
                h_hyb.coo_nnz++;
            }
        }
    }

    return h_hyb;
}

HYBDevice hyb_host_to_device(const HYBHost &h_hyb)
{
    HYBDevice d_hyb;
    d_hyb.nrows = h_hyb.nrows;
    d_hyb.ncols = h_hyb.ncols;
    d_hyb.ell_cols = h_hyb.ell_cols;
    d_hyb.coo_nnz = h_hyb.coo_nnz;

    const size_t ell_total = (size_t)h_hyb.ell_cols * h_hyb.nrows;

    // ELL
    cudaMalloc(&d_hyb.ell_col_idx, ell_total * sizeof(int));
    cudaMalloc(&d_hyb.ell_values, ell_total * sizeof(float));
    cudaMemcpy(d_hyb.ell_col_idx, h_hyb.ell_col_idx.data(),
               ell_total * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_hyb.ell_values, h_hyb.ell_values.data(),
               ell_total * sizeof(float), cudaMemcpyHostToDevice);

    // COO (may be empty)
    d_hyb.coo_row = nullptr;
    d_hyb.coo_col = nullptr;
    d_hyb.coo_val = nullptr;

    if (h_hyb.coo_nnz > 0)
    {
        cudaMalloc(&d_hyb.coo_row, h_hyb.coo_nnz * sizeof(int));
        cudaMalloc(&d_hyb.coo_col, h_hyb.coo_nnz * sizeof(int));
        cudaMalloc(&d_hyb.coo_val, h_hyb.coo_nnz * sizeof(float));
        cudaMemcpy(d_hyb.coo_row, h_hyb.coo_row.data(),
                   h_hyb.coo_nnz * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_hyb.coo_col, h_hyb.coo_col.data(),
                   h_hyb.coo_nnz * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_hyb.coo_val, h_hyb.coo_val.data(),
                   h_hyb.coo_nnz * sizeof(float), cudaMemcpyHostToDevice);
    }

    return d_hyb;
}

void hyb_device_free(HYBDevice &d_hyb)
{
    if (d_hyb.ell_col_idx)
    {
        cudaFree(d_hyb.ell_col_idx);
        d_hyb.ell_col_idx = nullptr;
    }
    if (d_hyb.ell_values)
    {
        cudaFree(d_hyb.ell_values);
        d_hyb.ell_values = nullptr;
    }
    if (d_hyb.coo_row)
    {
        cudaFree(d_hyb.coo_row);
        d_hyb.coo_row = nullptr;
    }
    if (d_hyb.coo_col)
    {
        cudaFree(d_hyb.coo_col);
        d_hyb.coo_col = nullptr;
    }
    if (d_hyb.coo_val)
    {
        cudaFree(d_hyb.coo_val);
        d_hyb.coo_val = nullptr;
    }
}

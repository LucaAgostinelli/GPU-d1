#pragma once
#include <vector>
#include <string>

struct COO
{
    int row, col;
    float val;
};

// ============================================================
// CSR format
// ============================================================

struct CSRHost
{
    int nrows, ncols, nnz;
    std::vector<int> row_ptr;
    std::vector<int> col_idx;
    std::vector<float> values;
};

struct CSRDevice
{
    int nrows, ncols, nnz;
    int *row_ptr;
    int *col_idx;
    float *values;
};

// ============================================================
// ELL format
// ============================================================
//
// Layout: column-major (for coalesced GPU access)
//   col_idx[c * nrows + row] = column index of the c-th NNZ in row
//   values [c * nrows + row] = value of the c-th NNZ in row
//
// Padding: rows shorter than max_col are padded with
//   col_idx = 0 (or any valid column, multiplied by 0)
//   values = 0.0f
//
// max_col: maximum number of NNZs across all rows
//   NOTE: for highly irregular matrices this can be very large,
//   wasting memory. Callers should check ell_fill_ratio() first.
//
struct ELLHost
{
    int nrows, ncols;
    int max_col;               // columns in the padded grid
    std::vector<int> col_idx;  // [max_col * nrows], column-major
    std::vector<float> values; // [max_col * nrows], column-major
};

struct ELLDevice
{
    int nrows, ncols;
    int max_col;
    int *col_idx;  // [max_col * nrows], column-major, device ptr
    float *values; // [max_col * nrows], column-major, device ptr
};

// ============================================================
// HYB format
// ============================================================
struct HYBHost
{
    int nrows, ncols;
    int ell_cols; // K - columns in the ELL part

    // ELL part  (column-major, size = ell_cols * nrows)
    std::vector<int> ell_col_idx;
    std::vector<float> ell_values;

    // COO part  (size = coo_nnz, sorted by row)
    int coo_nnz;
    std::vector<int> coo_row;
    std::vector<int> coo_col;
    std::vector<float> coo_val;
};

struct HYBDevice
{
    int nrows, ncols;
    int ell_cols;

    // ELL
    int *ell_col_idx;
    float *ell_values;

    // COO
    int coo_nnz;
    int *coo_row;
    int *coo_col;
    float *coo_val;
};

// ============================================================
// Functions
// ============================================================

std::vector<COO> read_mtx(const std::string &filename,
                          int &nrows, int &ncols,
                          bool &symmetric);

CSRHost coo_to_csr(std::vector<COO> &coo, int nrows, int ncols);

CSRDevice csr_host_to_device(const CSRHost &h_csr);

// Build ELL from CSR (host side).
// Returns the fill ratio  nnz_ell / nnz_real  so the caller can
// decide whether ELL is memory-efficient enough to use.
ELLHost csr_to_ell(const CSRHost &h_csr, float &fill_ratio);

ELLDevice ell_host_to_device(const ELLHost &h_ell);

void ell_device_free(ELLDevice &d_ell);

// Build HYB from CSR (host side)
HYBHost csr_to_hyb(const CSRHost &h_csr, int &ell_cols_out);

HYBDevice hyb_host_to_device(const HYBHost &h_hyb);

void hyb_device_free(HYBDevice &d_hyb);
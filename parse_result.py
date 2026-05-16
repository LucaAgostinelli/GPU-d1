#!/usr/bin/env python3
"""
Parse all .out files in outputs/ (or in the directory passed as an argument)
and produces CSV with SpMV benchmark results.

Expected RESULT_CSV format (13 fields):
RESULT_CSV,<path>,<nrows>,<ncols>,<nnz>,<kernel>,
<avg_ms>,<min_ms>,<max_ms>,<variance_ms>,<gflops>,<eff_bw_gbs>,<pct_peak>

Usage:
    python3 parse_results.py [outputs_dir] [--out results.csv]

Outputs:
    results.csv                  -- one line per (matrix, kernel)
    results_pivot_gflops.csv     -- matrices as rows, kernels as columns, GFLOP/s
    results_pivot_ms.csv         -- ditto, avg_ms
    results_pivot_bw.csv         -- ditto, eff_bw_gbs
    results_pivot_variance.csv   -- ditto, variance_ms
"""

import os
import sys
import csv
import glob
import argparse
from pathlib import Path
from collections import defaultdict


HEADER = [
    "matrix", "nrows", "ncols", "nnz", "avg_nnz_row",
    "kernel",
    "avg_ms", "min_ms", "max_ms", "variance_ms",
    "gflops", "eff_bw_gbs", "pct_peak",
]


def parse_file(filepath):
    result_rows = []

    with open(filepath, "r", errors="replace") as f:
        for line in f:
            line = line.strip()

            if not line.startswith("RESULT_CSV,"):
                continue

            parts = line.split(",")

            if len(parts) == 13:
                (_, mat_path, nrows, ncols, nnz, kernel,
                 avg_ms, min_ms, max_ms, variance_ms,
                 gflops, eff_bw_gbs, pct_peak) = parts
            elif len(parts) == 12:
                # Legacy files without variance column — set variance to None
                (_, mat_path, nrows, ncols, nnz, kernel,
                 avg_ms, min_ms, max_ms,
                 gflops, eff_bw_gbs, pct_peak) = parts
                variance_ms = None
            else:
                print(
                    f"  [WARN] unexpected RESULT_CSV field count "
                    f"({len(parts)}) in {filepath}: {line}",
                    file=sys.stderr,
                )
                continue

            matrix_name = Path(mat_path).stem
            nrows_i = int(nrows)
            nnz_i = int(nnz)
            avg_nnz = round(nnz_i / nrows_i, 2) if nrows_i > 0 else 0.0

            result_rows.append({
                "matrix":       matrix_name,
                "nrows":        nrows_i,
                "ncols":        int(ncols),
                "nnz":          nnz_i,
                "avg_nnz_row":  avg_nnz,
                "kernel":       kernel,
                "avg_ms":       float(avg_ms),
                "min_ms":       float(min_ms),
                "max_ms":       float(max_ms),
                "variance_ms":  float(variance_ms) if variance_ms is not None else None,
                "gflops":       float(gflops),
                "eff_bw_gbs":   float(eff_bw_gbs) if eff_bw_gbs != "N/A" else None,
                "pct_peak":     float(pct_peak) if pct_peak != "N/A" else None,
            })

    return result_rows


def _fmt(val, decimals=4):
    if val is None:
        return "N/A"
    return f"{val:.{decimals}f}"


def _write_pivot(rows, path, value_col, decimals=4):
    kernels = sorted({r["kernel"] for r in rows})
    by_matrix = defaultdict(dict)
    meta = {}
    for r in rows:
        m = r["matrix"]
        by_matrix[m][r["kernel"]] = r[value_col]
        meta[m] = (r["nrows"], r["ncols"], r["nnz"], r["avg_nnz_row"])

    fieldnames = ["matrix", "nrows", "ncols", "nnz", "avg_nnz_row"] + kernels
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for m in sorted(by_matrix.keys()):
            nrows, ncols, nnz, avg = meta[m]
            row = {
                "matrix": m,
                "nrows": nrows,
                "ncols": ncols,
                "nnz": nnz,
                "avg_nnz_row": avg,
            }
            for k in kernels:
                row[k] = _fmt(by_matrix[m].get(k), decimals)
            writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(
        description="Parse SpMV benchmark .out files into CSV tables.")
    parser.add_argument(
        "outputs_dir", nargs="?", default="outputs",
        help="Directory containing .out files (default: outputs/)")
    parser.add_argument(
        "--out", default="results.csv",
        help="Output CSV filename (default: results.csv)")
    args = parser.parse_args()

    out_dir = args.outputs_dir
    if not os.path.isdir(out_dir):
        print(f"Error: directory '{out_dir}' not found.", file=sys.stderr)
        sys.exit(1)

    out_files = sorted(glob.glob(os.path.join(out_dir, "*.out")))
    if not out_files:
        print(f"No .out files found in '{out_dir}'.", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(out_files)} .out file(s) in '{out_dir}'")

    all_result_rows = []

    for f in out_files:
        r_rows = parse_file(f)
        print(
            f"  {os.path.basename(f):40s}  -> {len(r_rows)} RESULT_CSV rows")
        all_result_rows.extend(r_rows)

    if not all_result_rows:
        print(
            "No RESULT_CSV lines found. Rebuild and re-run the benchmark.",
            file=sys.stderr)
        sys.exit(1)

    all_result_rows.sort(key=lambda r: (r["matrix"], r["kernel"]))

    csv_dir = os.path.join(os.getcwd(), "csv")
    os.makedirs(csv_dir, exist_ok=True)

    # ---- Flat results CSV ----
    flat_path = os.path.join(csv_dir, args.out)
    with open(flat_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=HEADER)
        writer.writeheader()
        for r in all_result_rows:
            writer.writerow({
                **r,
                "variance_ms": _fmt(r["variance_ms"], 9),
                "eff_bw_gbs":  _fmt(r["eff_bw_gbs"]),
                "pct_peak":    _fmt(r["pct_peak"], 2),
            })
    print(f"\nFlat results CSV:                {flat_path}  "
          f"({len(all_result_rows)} rows)")

    # ---- Pivot CSVs ----
    base = flat_path.replace(".csv", "")

    p_gf = f"{base}_pivot_gflops.csv"
    _write_pivot(all_result_rows, p_gf, "gflops")
    print(f"Pivot GFLOP/s written to:        {p_gf}")

    p_ms = f"{base}_pivot_ms.csv"
    _write_pivot(all_result_rows, p_ms, "avg_ms")
    print(f"Pivot avg_ms  written to:        {p_ms}")

    p_bw = f"{base}_pivot_bw.csv"
    _write_pivot(all_result_rows, p_bw, "eff_bw_gbs")
    print(f"Pivot eff_bw  written to:        {p_bw}")

    p_var = f"{base}_pivot_variance.csv"
    _write_pivot(all_result_rows, p_var, "variance_ms", 9)
    print(f"Pivot variance written to:       {p_var}")


if __name__ == "__main__":
    main()

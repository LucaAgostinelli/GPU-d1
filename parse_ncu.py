#!/usr/bin/env python3
"""
Extract Nsight Compute metrics from .txt (--csv) output produced by ncu.sh
and aggregate them into report-ready CSVs.

UNITS (read directly from ncu's "Metric Unit" column, no assumptions):
  gpu__time_duration.sum          -> "usecond"  -> stored as-is in kernel_time_us
  dram__bytes_read/write.sum      -> "Mbyte" or "Kbyte" etc -> converted to MB
  smsp__inst_executed_*           -> "inst" (raw count, comma-thousands) -> millions
  hit rates, coalescing, occupancy -> "%"  -> stored as-is
  sectors_per_load_req            -> "sector/request" -> stored as-is
"""

import os
import sys
import csv
import glob
import argparse
from collections import defaultdict
from pathlib import Path

METRIC_MAP = {
    "l1tex__t_sector_hit_rate.pct":
        "l1_hit_pct",
    "lts__t_sector_hit_rate.pct":
        "l2_hit_pct",
    "smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct":
        "coalescing_pct",
    "smsp__warps_active.avg.pct_of_peak_sustained_active":
        "warp_occ_pct",
    "gpu__time_duration.sum":
        "kernel_time_us",
    "dram__bytes_read.sum":
        "dram_read_mb",
    "dram__bytes_write.sum":
        "dram_write_mb",
    "l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio":
        "sectors_per_load_req",
    "smsp__inst_executed_op_global_ld.sum":
        "global_loads_M",
    "smsp__inst_executed_op_global_st.sum":
        "global_stores_M",
}

FLAT_COLUMNS = [
    "matrix", "kernel",
    "l1_hit_pct", "l2_hit_pct", "coalescing_pct", "warp_occ_pct",
    "kernel_time_us",
    "dram_read_mb", "dram_write_mb", "dram_total_mb",
    "sectors_per_load_req",
    "global_loads_M", "global_stores_M",
]

_DRAM_COLS = {"dram_read_mb", "dram_write_mb"}
_TIME_COLS = {"kernel_time_us"}
_INST_COLS = {"global_loads_M", "global_stores_M"}


def _safe_float(s: str):
    """
    Parse a numeric string from ncu --csv output.
    """
    s = s.strip().strip('"')
    if not s or s in ("N/A", "n/a", "-"):
        return None

    # Detect thousands separators: a comma followed by exactly 3 digits,
    # either at end of string or before another comma/digits block.
    # Strategy: if ALL commas are followed by exactly 3 word-chars and then
    # either end-of-string or another comma, treat them all as thousands seps.
    import re
    if re.fullmatch(r'\d{1,3}(,\d{3})*', s):
        # Pure integer with thousands separators: "294,173" or "3,489,366"
        s = s.replace(',', '')
    elif ',' in s and '.' not in s:
        # Single comma, not matching thousands pattern -> European decimal
        s = s.replace(',', '.')
    # else: already uses dot as decimal (normal English float), leave as-is

    try:
        return float(s)
    except ValueError:
        return None


def _convert(value: float, unit: str, col: str) -> float:
    u = unit.lower().strip()

    if col in _DRAM_COLS:
        if u in ("byte", "bytes"):
            return value / 1e6
        if u in ("kbyte", "kbytes", "kb"):
            return value / 1e3
        if u in ("mbyte", "mbytes", "mb"):
            return value
        if u in ("gbyte", "gbytes", "gb"):
            return value * 1e3
        print(f"  [WARN] unknown unit '{unit}' for {col}, assuming bytes",
              file=sys.stderr)
        return value / 1e6

    if col in _TIME_COLS:
        if u in ("usecond", "us", "microsecond", "microseconds"):
            return value
        if u in ("nsecond", "ns", "nanosecond", "nanoseconds"):
            return value / 1e3
        if u in ("msecond", "ms", "millisecond", "milliseconds"):
            return value * 1e3
        print(f"  [WARN] unknown unit '{unit}' for {col}, assuming usecond",
              file=sys.stderr)
        return value

    if col in _INST_COLS:
        if u in ("inst", "instruction", "instructions", ""):
            return value / 1e6
        if u in ("kinst", "k"):
            return value / 1e3
        if u in ("minst", "m"):
            return value
        print(f"  [WARN] unknown unit '{unit}' for {col}, assuming raw inst",
              file=sys.stderr)
        return value / 1e6

    # Percentages ("%"), ratios ("sector/request"): return as-is
    return value


# ---------------------------------------------------------------------------
# File parsing
# ---------------------------------------------------------------------------

def parse_ncu_csv_txt(filepath: str) -> dict:
    metrics: dict[str, float] = {}
    try:
        with open(filepath, "r", errors="replace") as f:
            content = f.read()
    except OSError:
        return metrics

    lines = content.splitlines()
    header_idx = None
    for i, line in enumerate(lines):
        if "Metric Name" in line:
            header_idx = i
            break
    if header_idx is None:
        return metrics

    reader = csv.DictReader(
        (l.strip() for l in lines[header_idx:]),
        quotechar='"',
    )
    for row in reader:
        metric_name = (row.get("Metric Name") or "").strip().strip('"')
        metric_val = (row.get("Metric Value") or "").strip().strip('"')
        metric_unit = (row.get("Metric Unit") or "").strip().strip('"')

        if metric_name not in METRIC_MAP:
            continue
        val = _safe_float(metric_val)
        if val is None:
            continue
        col = METRIC_MAP[metric_name]
        metrics[col] = _convert(val, metric_unit, col)

    return metrics


# ---------------------------------------------------------------------------
# Filename -> (matrix_name, kernel_tag)
# ---------------------------------------------------------------------------

KNOWN_KERNELS = [
    "S-V-ADAPTIVE_scalar", "S-V-ADAPTIVE_vector",
    "FLAT_prep", "FLAT_core",
    "HYB_ell", "HYB_coo",
    "SCALAR", "LINE", "ELL", "cuSPARSE",
]


def filename_to_meta(fname: str):
    stem = Path(fname).stem.replace("_metrics", "")
    if "__" in stem:
        parts = stem.split("__", 1)
        return parts[0], parts[1]
    for k in sorted(KNOWN_KERNELS, key=len, reverse=True):
        if stem.endswith("_" + k):
            return stem[: -(len(k) + 1)], k
    parts = stem.rsplit("_", 1)
    return (parts[0], parts[1]) if len(parts) == 2 else (stem, "UNKNOWN")


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def _fmt(val, dec=2):
    return "N/A" if val is None else f"{val:.{dec}f}"


def write_pivot(rows, out_path, value_col, dec=2):
    kernels = sorted({r["kernel"]
                     for r in rows if r.get(value_col) is not None})
    matrices = sorted({r["matrix"] for r in rows})
    by: dict = defaultdict(dict)
    for r in rows:
        if r.get(value_col) is not None:
            by[r["matrix"]][r["kernel"]] = r[value_col]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["matrix"] + kernels)
        writer.writeheader()
        for m in matrices:
            row = {"matrix": m}
            for k in kernels:
                row[k] = _fmt(by[m].get(k), dec)
            writer.writerow(row)


def print_ascii_table(rows, col, label, dec=2):
    kernels = sorted({r["kernel"] for r in rows})
    matrices = sorted({r["matrix"] for r in rows})
    by: dict = defaultdict(dict)
    for r in rows:
        if r.get(col) is not None:
            by[r["matrix"]][r["kernel"]] = r[col]
    kw = 16
    header = f"{'Matrix':<22}" + "".join(f"{k:>{kw}}" for k in kernels)
    sep = "=" * len(header)
    print(f"\n{sep}\n{label}\n{sep}\n{header}\n{'-'*len(header)}")
    for m in matrices:
        line = f"{m:<22}" + \
            "".join(f"{_fmt(by[m].get(k), dec):>{kw}}" for k in kernels)
        print(line)
    print(sep)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ncu_dir", nargs="?", default="outputs/ncu")
    ap.add_argument("--out", default="ncu_results.csv")
    args = ap.parse_args()

    if not os.path.isdir(args.ncu_dir):
        print(f"Error: '{args.ncu_dir}' not found.", file=sys.stderr)
        sys.exit(1)

    txt_files = sorted(glob.glob(os.path.join(args.ncu_dir, "*_metrics.txt")))
    if not txt_files:
        txt_files = sorted(glob.glob(os.path.join(args.ncu_dir, "*.txt")))
    if not txt_files:
        print(f"No metric files found in '{args.ncu_dir}'.", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(txt_files)} metric file(s) in '{args.ncu_dir}'\n")

    all_rows = []
    for fpath in txt_files:
        mat, kernel = filename_to_meta(os.path.basename(fpath))
        metrics = parse_ncu_csv_txt(fpath)
        if not metrics:
            print(f"  (no data)   {os.path.basename(fpath)}")
            continue

        rd = metrics.get("dram_read_mb")
        wr = metrics.get("dram_write_mb")
        metrics["dram_total_mb"] = (
            (rd or 0.0) + (wr or 0.0)
            if (rd is not None or wr is not None) else None
        )

        row: dict = {"matrix": mat, "kernel": kernel}
        for col in FLAT_COLUMNS[2:]:
            row[col] = metrics.get(col)
        all_rows.append(row)

        print(f"  {mat:<20s}  {kernel:<16s}  "
              f"L1={_fmt(metrics.get('l1_hit_pct'))}%  "
              f"L2={_fmt(metrics.get('l2_hit_pct'))}%  "
              f"coal={_fmt(metrics.get('coalescing_pct'))}%  "
              f"occ={_fmt(metrics.get('warp_occ_pct'))}%  "
              f"DRAM={_fmt(metrics.get('dram_total_mb'))} MB  "
              f"time={_fmt(metrics.get('kernel_time_us'))} us  "
              f"loads={_fmt(metrics.get('global_loads_M'), 3)}M")

    if not all_rows:
        print("\nNo metrics extracted.", file=sys.stderr)
        sys.exit(1)

    all_rows.sort(key=lambda r: (r["matrix"], r["kernel"]))

    csv_dir = os.path.join(os.getcwd(), "csv")
    os.makedirs(csv_dir, exist_ok=True)

    flat_path = os.path.join(csv_dir, args.out)
    with open(flat_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FLAT_COLUMNS)
        writer.writeheader()
        for r in all_rows:
            writer.writerow({
                c: (_fmt(r[c], 4) if isinstance(
                    r[c], float) else (r[c] or "N/A"))
                for c in FLAT_COLUMNS
            })
    print(f"\nFlat CSV:  {flat_path}  ({len(all_rows)} rows)")

    base = flat_path.replace(".csv", "")
    pivots = [
        ("l1_hit_pct",           "pivot_l1hit",
         "L1 TEX hit-rate (%)",                    2),
        ("l2_hit_pct",           "pivot_l2hit",
         "L2 hit-rate (%)",                        2),
        ("coalescing_pct",       "pivot_coal",
         "Global-load coalescing efficiency (%)",   2),
        ("warp_occ_pct",         "pivot_occ",
         "Warp occupancy (%)",                     2),
        ("dram_total_mb",        "pivot_dram",
         "DRAM total traffic (MB)",                2),
        ("sectors_per_load_req", "pivot_sectors",
         "Sectors per load request (ideal = 1.0)", 3),
        ("kernel_time_us",       "pivot_time",
         "Kernel time (us, NCU overhead included)", 2),
        ("global_loads_M",       "pivot_loads",
         "Global load instructions (millions)",    3),
    ]
    for col, suffix, label, dec in pivots:
        p = f"{base}_{suffix}.csv"
        write_pivot(all_rows, p, col, dec)
        print(f"  {p}")
        print_ascii_table(all_rows, col, label, dec)

    print("\nDone.")


if __name__ == "__main__":
    main()

#!/bin/bash
# =============================================================================
# Extract metrics from .ncu-rep files into parseable CSVs.
#
# Usage:
#   bash ncu_extract.sh [ncu_dir]
#   ncu_dir defaults to outputs/ncu
# =============================================================================

set -euo pipefail

NCU_DIR="${1:-outputs/ncu}"

if [[ ! -d "$NCU_DIR" ]]; then
    echo "Error: directory '$NCU_DIR' not found." >&2
    exit 1
fi

module load CUDA/11.8.0 2>/dev/null || true

METRICS=(
    "l1tex__t_sector_hit_rate.pct"
    "lts__t_sector_hit_rate.pct"
    "smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct"
    "smsp__warps_active.avg.pct_of_peak_sustained_active"
    "gpu__time_duration.sum"
    "dram__bytes_read.sum"
    "dram__bytes_write.sum"
    "l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio"
    "smsp__inst_executed_op_global_ld.sum"
    "smsp__inst_executed_op_global_st.sum"
)
METRIC_LIST=$(IFS=,; echo "${METRICS[*]}")

rep_files=( "$NCU_DIR"/*.ncu-rep )

if [[ ${#rep_files[@]} -eq 0 ]] || [[ ! -f "${rep_files[0]}" ]]; then
    echo "No .ncu-rep files found in '$NCU_DIR'." >&2
    exit 1
fi

echo "Found ${#rep_files[@]} .ncu-rep file(s) in '$NCU_DIR'"

for rep in "${rep_files[@]}"; do
    # Input:  e.g. outputs/ncu/roadNet-CA__SCALAR.ncu-rep
    # Output: e.g. outputs/ncu/roadNet-CA__SCALAR_metrics.txt
    stem="${rep%.ncu-rep}"
    out="${stem}_metrics.txt"
    name=$(basename "$rep")

    echo -n "  $name -> "

    ncu --import "$rep" \
        --csv \
        --metrics "$METRIC_LIST" \
        > "$out" 2>&1

    if grep -q '"Metric Name"' "$out" 2>/dev/null; then
        echo "ok"
    else
        echo "WARNING: no metric data in output (kernel may not have run)"
    fi
done

echo ""
echo "=== Extraction complete. Run:  python3 parse_ncu.py  ==="
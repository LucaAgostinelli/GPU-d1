#!/bin/bash
# =============================================================================
# Nsight Compute profiling for SpMV memory-behavior analysis
#
# Called by ncu_sbatch.sh with the SLURM array task index as $1.
# Each invocation profiles one matrix against all kernel tags.
#
# Usage
# -----
#   bash ncu.sh <task_id>
#   sbatch ncu_sbatch.sh
#
# Task index -> matrix mapping:
#   0  -> amazon0302
#   1  -> ASIC_680k
#   2  -> cage13
#   3  -> cit-Patents
#   4  -> crankseg_2
#   5  -> parabolic_fem
#   6  -> poisson3Da
#   7  -> roadNet-CA
#   8  -> thermal2
#   9  -> webbase-1M
#
# Output  ->  outputs/ncu/<matrix>_<kernel>.{ncu-rep,txt}
# =============================================================================

set -euo pipefail

TASK_ID="${1:?Usage: ncu.sh <task_id>}"
BINARY="./bin/spmv"
OUT_DIR="outputs/ncu"
mkdir -p "$OUT_DIR"

MAT_NAMES=(
    "amazon0302"
    "ASIC_680k"
    "cage13"
    "cit-Patents"
    "crankseg_2"
    "parabolic_fem"
    "poisson3Da"
    "roadNet-CA"
    "thermal2"
    "webbase-1M"
)
MATRICES=(
    "/home/luca.agostinelli/Deliverable_1/matrices/amazon0302.mtx"
    "/home/luca.agostinelli/Deliverable_1/matrices/ASIC_680k.mtx"
    "/home/luca.agostinelli/Deliverable_1/matrices/cage13.mtx"
    "/home/luca.agostinelli/Deliverable_1/matrices/cit-Patents.mtx"
    "/home/luca.agostinelli/Deliverable_1/matrices/crankseg_2.mtx"
    "/home/luca.agostinelli/Deliverable_1/matrices/parabolic_fem.mtx"
    "/home/luca.agostinelli/Deliverable_1/matrices/poisson3Da.mtx"
    "/home/luca.agostinelli/Deliverable_1/matrices/roadNet-CA.mtx"
    "/home/luca.agostinelli/Deliverable_1/matrices/thermal2.mtx"
    "/home/luca.agostinelli/Deliverable_1/matrices/webbase-1M.mtx"
)

mat_name="${MAT_NAMES[$TASK_ID]}"
mat_path="${MATRICES[$TASK_ID]}"

echo "Task ${TASK_ID}: profiling matrix '${mat_name}'"
echo "  path: ${mat_path}"

# ---------------------------------------------------------------------------
# Kernels to profile.
# TAG        : used as filename suffix  ->  outputs/ncu/<matrix>_<TAG>.ncu-rep
# REGEX      : matched against the CUDA kernel function name by ncu
# ---------------------------------------------------------------------------
KERNEL_TAGS=(
    "SCALAR"
    "S-V-ADAPTIVE_scalar"
    "S-V-ADAPTIVE_vector"
    "LINE"
    "FLAT_prep"
    "FLAT_core"
    "ELL"
    "HYB_ell"
    "HYB_coo"
    "cuSPARSE"
)
KERNEL_REGEXES=(
    "spmv_scalar"
    "spmv_csr_scalar"
    "spmv_csr_vector"
    "spmv_line_enhance"
    "flat_preprocess"
    "spmv_flat"
    "spmv_ell_kernel"
    "spmv_ell_kernel"  
    "spmv_hyb_coo"
    "cusparse"
)

# ---------------------------------------------------------------------------
# Metrics.
# ---------------------------------------------------------------------------
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

# Use a private TMPDIR so ncu creates its lock file in a user-owned dir,
# avoiding permission clashes with other jobs on the same node.
export TMPDIR="/tmp/${USER}_ncu_$$"
mkdir -p "$TMPDIR"

for i in "${!KERNEL_TAGS[@]}"; do
    kernel_tag="${KERNEL_TAGS[$i]}"
    kernel_regex="${KERNEL_REGEXES[$i]}"

    rep_file="${OUT_DIR}/${mat_name}__${kernel_tag}.ncu-rep"
    txt_file="${OUT_DIR}/${mat_name}__${kernel_tag}.txt"

    echo "  [${kernel_tag}] regex='${kernel_regex}'"

    ncu \
        --target-processes all \
        --kernel-name regex:"${kernel_regex}" \
        --launch-skip 20 \
        --launch-count 1 \
        --metrics "${METRIC_LIST}" \
        --export "${rep_file}" \
        --csv \
        "${BINARY}" "${mat_path}" \
        > "${txt_file}" 2>&1

    echo "    -> ${txt_file}"
done

rm -rf "$TMPDIR"

echo ""
echo "=== Task ${TASK_ID} (${mat_name}) complete ==="
echo "=== Run  python3 parse_ncu.py  after all tasks finish ==="
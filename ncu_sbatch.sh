#!/bin/bash
# =============================================================================
# SLURM array job for Nsight Compute profiling
# One task per matrix (10 matrices total).
# Node edu01 is used because other nodes do not allow ncu to run.
#
# Usage:  sbatch ncu_sbatch.sh
#
# Task index -> matrix mapping (must match MAT_NAMES in ncu.sh):
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
# =============================================================================
#SBATCH --partition=edu-medium
#SBATCH --account=gpu.computing26
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:20:00
#SBATCH --nodelist=edu01

#SBATCH --job-name=spmv_ncu
#SBATCH --array=0-9
#SBATCH --output=outputs/ncu_%A_%a.out
#SBATCH --error=outputs/ncu_%A_%a.err

set -euo pipefail

module load CUDA/11.8.0

mkdir -p outputs/ncu

bash ncu.sh "$SLURM_ARRAY_TASK_ID"
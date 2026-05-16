#!/bin/bash
#SBATCH --partition=edu-short
#SBATCH --account=gpu.computing26
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:05:00
#SBATCH --nodelist=edu01

#SBATCH --job-name=spmv_benchmark
#SBATCH --array=0-9
#SBATCH --output=outputs/%x-%A_%a.out
#SBATCH --error=outputs/%x-%A_%a.err

set -e

module load CUDA/11.8.0

mkdir -p outputs

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

matrix="${MATRICES[$SLURM_ARRAY_TASK_ID]}"
matrix_name=$(basename "$matrix" .mtx)

echo "Running matrix: $matrix_name"
./bin/spmv "$matrix"
# ./tests/test_parser
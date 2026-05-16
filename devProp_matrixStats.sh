#!/bin/bash
#SBATCH --partition=edu-short
#SBATCH --account=gpu.computing26
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:05:00
#SBATCH --nodelist=edu01

#SBATCH --job-name=matrix_stats
#SBATCH --output=outputs/dev_prop_matrix_stats-%j.out
#SBATCH --error=outputs/dev_prop_matrix_stats-%j.err

set -e

module load CUDA/11.8.0

mkdir -p outputs
mkdir -p csv

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

OUTPUT_FILE="csv/matrix_stats.csv"

# CSV header
echo "type,path,nrows,ncols,nnz,symmetric,parse_time,avg_nnz,min_row_nnz,max_row_nnz,std_nnz,imbalance" > "$OUTPUT_FILE"

echo "================ GPU INFO ================"
./bin/printDevProp
echo "=========================================="

echo "Running matrix statistics..."

for MATRIX in "${MATRICES[@]}"
do
    echo "Processing: $MATRIX" >&2

    ./bin/matrix_stats "$MATRIX" >> "$OUTPUT_FILE"
done

echo "DONE"
echo "CSV saved to: $OUTPUT_FILE"
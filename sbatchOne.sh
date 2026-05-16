#!/bin/bash
#SBATCH --partition=edu-medium
#SBATCH --account=gpu.computing26
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:20:00
#SBATCH --nodelist=edu01

#SBATCH --job-name=spmv_benchmark
#SBATCH --output=outputs/%x-%j.out
#SBATCH --error=outputs/%x-%j.err

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

KERNELS=(
    "SCALAR"
    "S-V-ADAPTIVE"
    "LINE"
    "FLAT"
    "ACC"
    "ELL"
    "HYB"
    "cuSPARSE"
)

for matrix in "${MATRICES[@]}"; do
    matrix_name=$(basename "$matrix" .mtx)

    for kernel in "${KERNELS[@]}"; do
        echo "Running matrix=${matrix_name}  kernel=${kernel}"

        ./bin/spmv "$matrix" "$kernel" \
            >> "outputs/${matrix_name}.out" \
            2>> "outputs/${matrix_name}.err"
    done

    echo "Done: $matrix_name"
done
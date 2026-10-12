#!/bin/bash
#SBATCH --partition=compute
#SBATCH --mem=12G
#SBATCH -n 4
#SBATCH --time=02:00:00
#SBATCH --output=Output/analyse_post_hoc_hardness.test.out

module load miniforge
mamba activate pytorch

python -m src.experiments.analyse_post_hoc_hardness --dataset_name CIFAR-100
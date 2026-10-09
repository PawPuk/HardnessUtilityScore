#!/bin/bash
#SBATCH --mem=12G
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=24:00:00
#SBATCH --output=Output/estimate_hardness_post_hoc_on_CIFAR100.test.out

# Load the modules required by our program
module load miniforge
source activate pytorch

python3 -m src.experiments.estimate_hardness_post_hoc --dataset_name CIFAR-100


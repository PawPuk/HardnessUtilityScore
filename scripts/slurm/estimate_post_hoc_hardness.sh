#!/bin/bash
#SBATCH --mem=12G
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --output=Output/estimate_hardness_post_hoc_on_CIFAR100.test.out

# Load the modules required by our program
module load miniforge
module load CUDA/10.2.89-GCC-8.3.0
source activate pytorch

python3 -m src.experiments.estimate_hardness_post_hoc --dataset_name CIFAR-100


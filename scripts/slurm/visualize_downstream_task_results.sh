#!/bin/bash
#SBATCH --mem=16G
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=10:00:00
#SBATCH --output=Output/visualize_HBR_on_CIFAR100.test.out

# Load the modules required by our program
module load Anaconda3/2022.05
module load CUDA/10.2.89-GCC-8.3.0
source activate pytorch

python3 -m src.experiments.visualize_downstream_task_results --dataset_name CIFAR-100


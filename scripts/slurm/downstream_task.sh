#!/bin/bash
#SBATCH --partition=gpushort
#SBATCH --nodes=8
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-cpu=11G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=Output/edm_downstream_task_on_CIFAR100.test.out

module load miniforge
mamba activate pytorch

python -m src.experiments.downstream_task --dataset_name CIFAR-100 --generative_architecture edm
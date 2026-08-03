#!/bin/bash

set -e

generative_models=("edm" "vae" "stylegan")
oversampling_strategies=("hard" "random")

for generative_model in "${generative_models[@]}"
do
    for oversampling_strategy in "${oversampling_strategies[@]}"
    do
        if [ "${oversampling_strategy}" == "hard" ]; then
            suffix="h"
        else
            suffix="r"
        fi

        job_name="${suffix}${generative_model}"
        log_file="Output/output_hbr_${oversampling_strategy}_${generative_model}_CIFAR100_alpha_2.0.out"

        sbatch \
            --job-name="$job_name" \
            --output="$log_file" \
            scripts/slurm/downstream_task.sh "CIFAR-100" \
            "$generative_model" "$oversampling_strategy" | awk '{print $4}'

    done
done

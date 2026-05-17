#!/bin/bash
set -euo pipefail

if [ "$#" -lt 2 ]; then
  echo "usage: bash scripts/slurm/submit_dependent_pipeline.sh <job1_id> <job2_scifact_id>"
  echo "example: bash scripts/slurm/submit_dependent_pipeline.sh 1566538 1566547"
  exit 1
fi

job1_id="$1"
job2_scifact_id="$2"

job2_labels_id=$(sbatch --dependency=afterok:${job1_id} scripts/slurm/job2_build_labels.sbatch | awk '{print $4}')
job4_small_id=$(sbatch --dependency=afterok:${job1_id} scripts/slurm/job4_bottleneck_small.sbatch | awk '{print $4}')
job3_week9_id=$(sbatch --dependency=afterok:${job2_scifact_id}:${job2_labels_id} scripts/slurm/job3_week9_eval.sbatch | awk '{print $4}')
job3_scaling_id=$(sbatch --dependency=afterok:${job2_scifact_id}:${job2_labels_id} scripts/slurm/job3_scaling.sbatch | awk '{print $4}')
job4_full_id=$(sbatch --dependency=afterok:${job4_small_id} scripts/slurm/job4_bottleneck_full.sbatch | awk '{print $4}')

echo "Submitted dependent jobs:"
echo "  job2_labels: ${job2_labels_id} (afterok:${job1_id})"
echo "  job4_small:  ${job4_small_id} (afterok:${job1_id})"
echo "  job3_week9:  ${job3_week9_id} (afterok:${job2_scifact_id}:${job2_labels_id})"
echo "  job3_scale:  ${job3_scaling_id} (afterok:${job2_scifact_id}:${job2_labels_id})"
echo "  job4_full:   ${job4_full_id} (afterok:${job4_small_id})"

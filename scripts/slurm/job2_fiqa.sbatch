#!/bin/bash
#SBATCH --job-name=job2_fiqa
#SBATCH --partition=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/job2_fiqa_%j.out
#SBATCH --error=logs/job2_fiqa_%j.err

cd ~/Rodriguez-Capcha-CS197-M2
uv sync
export PYTHONPATH=$PWD
mkdir -p logs outputs outputs/beir_data

uv run python scripts/jobs/job2_fiqa_segments.py \
  --download \
  --data-dir outputs/beir_data \
  --output-path outputs/fiqa_sweep_records.json

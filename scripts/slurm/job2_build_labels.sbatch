#!/bin/bash
#SBATCH --job-name=job2_labels
#SBATCH --partition=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/job2_labels_%j.out
#SBATCH --error=logs/job2_labels_%j.err

cd ~/Rodriguez-Capcha-CS197-M2
uv sync
export PYTHONPATH=$PWD
mkdir -p logs outputs outputs/beir_data

uv run python scripts/jobs/job2_build_training_labels.py \
  --fineweb-queries-path outputs/fineweb_queries.jsonl \
  --marco-data-path outputs/beir_data/msmarco \
  --max-queries-for-sweep 25000 \
  --fineweb-output-path outputs/fineweb_labeled.json \
  --marco-output-path outputs/marco_labeled.json

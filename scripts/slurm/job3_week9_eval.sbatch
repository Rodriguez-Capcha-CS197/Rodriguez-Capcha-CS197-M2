#!/bin/bash
#SBATCH --job-name=job3_week9
#SBATCH --partition=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=logs/job3_week9_%j.out
#SBATCH --error=logs/job3_week9_%j.err

cd ~/Rodriguez-Capcha-CS197-M2
uv sync
export PYTHONPATH=$PWD
mkdir -p logs outputs/week9

uv run python scripts/jobs/job3_week9_eval.py \
  --fineweb-records outputs/fineweb_labeled.json \
  --marco-records outputs/marco_labeled.json \
  --scifact-records outputs/scifact_sweep_records.json \
  --fiqa-records outputs/fiqa_sweep_records.json \
  --scifact-data-path outputs/beir_data/scifact \
  --fiqa-data-path outputs/beir_data/fiqa \
  --output-dir outputs/week9

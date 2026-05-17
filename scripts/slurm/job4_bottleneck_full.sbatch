#!/bin/bash
#SBATCH --job-name=job4_full
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/job4_full_%j.out
#SBATCH --error=logs/job4_full_%j.err

cd ~/Rodriguez-Capcha-CS197-M2
module load cuda
uv sync
export PYTHONPATH=$PWD
mkdir -p logs outputs

uv run python scripts/jobs/job4_paper_bottleneck_ablation.py \
  --fineweb-queries-path outputs/fineweb_queries.jsonl \
  --max-fineweb-queries 25000 \
  --scifact-data-dir outputs/beir_data \
  --k-shot 10 \
  --output-path outputs/paper_bottleneck_ablation_scifact.json

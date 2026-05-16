# CS195S2026 Week 9 Pipeline

This repo now includes a 3-job pipeline for cross-domain information bottleneck testing.

## Install

```bash
uv sync
```

## Repository layout

- `shared`: shared data/eval/inference utilities
- `ska_agent`: SKA agent package and CLI
- `scripts/jobs`: reproducible batch jobs for each experiment stage
- `notebooks`: exploratory analysis notebooks
- `configs`: run/config manifests (intended location)
- `data`: local datasets (`raw`, `interim`, `processed`; gitignored)
- `artifacts`: generated reports/figures/checkpoints (gitignored)

## Job 1 (FineWeb query generation)

CPU fallback:

```bash
uv run scripts/jobs/job1_generate_queries.py --generator template --n-passages 10000 --n-queries 2
```

500M-token target (approximate):

```bash
uv run scripts/jobs/job1_generate_queries.py \
  --generator template \
  --target-tokens 500000000 \
  --n-passages 10000000 \
  --max-passages-for-queries 25000 \
  --n-queries 2
```

GPU/Qwen mode:

```bash
uv run scripts/jobs/job1_generate_queries.py --generator qwen --qwen-model Qwen/Qwen3.5-2B
```

Output:
- `outputs/fineweb_queries.jsonl`

## Job 2 (label generation + BEIR oracle sweeps)

Build training labels:

```bash
uv run scripts/jobs/job2_build_training_labels.py \
  --marco-data-path outputs/beir_data/msmarco \
  --max-queries-for-sweep 25000
```

Build SciFact oracle sweep:

```bash
uv run scripts/jobs/job2_scifact_segments.py --download
```

Build FiQA oracle sweep:

```bash
uv run scripts/jobs/job2_fiqa_segments.py --download
```

Outputs:
- `outputs/fineweb_labeled.json`
- `outputs/marco_labeled.json`
- `outputs/scifact_sweep_records.json`
- `outputs/fiqa_sweep_records.json`

## Job 3 (Week 9 evaluation)

```bash
uv run scripts/jobs/job3_week9_eval.py \
  --fineweb-records outputs/fineweb_labeled.json \
  --marco-records outputs/marco_labeled.json \
  --scifact-records outputs/scifact_sweep_records.json \
  --fiqa-records outputs/fiqa_sweep_records.json \
  --scifact-data-path outputs/beir_data/scifact \
  --fiqa-data-path outputs/beir_data/fiqa \
  --output-dir outputs/week9
```

Outputs:
- `outputs/week9/scifact_results.json`
- `outputs/week9/fiqa_results.json`
- `outputs/week9/week9_comparison_table.png`
- `outputs/week9/week9_comparison_table.pdf`

## Scaling sweep

```bash
uv run scripts/jobs/job3_scaling_sweep.py --output-dir outputs/week9/scaling
```

Outputs:
- `outputs/week9/scaling/scaling_summary.json`

## Notebook launcher

- `notebooks/week9_cross_robustness.ipynb`

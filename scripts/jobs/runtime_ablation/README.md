# Runtime Score-Gap Ablation

This folder contains an isolated Option A ablation job. It does not modify the
existing Week 9 evaluator or reporting files.

The job writes all new artifacts under:

```bash
outputs/runtime_score_gap_ablation
```

That directory is intended to be exported on its own.

## Required Restored Inputs

Restore these from the previous-output tarball before running:

```bash
outputs/fineweb_labeled.json
outputs/marco_labeled.json
outputs/scifact_sweep_records.json
outputs/fiqa_sweep_records.json
outputs/beir_data/scifact
outputs/beir_data/fiqa
```

## Run

```bash
PYTHONPATH=$PWD uv run python scripts/jobs/runtime_ablation/job3_runtime_score_gap_eval.py \
  --fineweb-records outputs/fineweb_labeled.json \
  --marco-records outputs/marco_labeled.json \
  --scifact-records outputs/scifact_sweep_records.json \
  --fiqa-records outputs/fiqa_sweep_records.json \
  --scifact-data-path outputs/beir_data/scifact \
  --fiqa-data-path outputs/beir_data/fiqa \
  --output-dir outputs/runtime_score_gap_ablation
```

## Outputs

```bash
outputs/runtime_score_gap_ablation/runtime_score_gap_runs.json
outputs/runtime_score_gap_ablation/scifact_runtime_score_gap_results.json
outputs/runtime_score_gap_ablation/fiqa_runtime_score_gap_results.json
outputs/runtime_score_gap_ablation/runtime_score_gap_policy_query_metrics.jsonl
outputs/runtime_score_gap_ablation/runtime_score_gap_summary.csv
outputs/runtime_score_gap_ablation/runtime_score_gap_summary.tex
outputs/runtime_score_gap_ablation/runtime_score_gap_bootstrap_summary.json
outputs/runtime_score_gap_ablation/runtime_score_gap_metadata.json
```

## Export Only This Ablation

```bash
tar -czf runtime_score_gap_ablation_outputs.tar.gz -C outputs runtime_score_gap_ablation
```

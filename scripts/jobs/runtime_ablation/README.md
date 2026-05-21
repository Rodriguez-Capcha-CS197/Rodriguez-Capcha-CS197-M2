# Runtime Score-Gap Ablation

This folder contains isolated runtime-aware ablation jobs. They do not modify
the existing Week 9 evaluator or reporting files.

## Existing Two-Domain Runtime+MLP Comparison

`job3_runtime_score_gap_eval.py` reuses the Week 9 MLP policies and adds the
runtime score-gap policy in a separate output directory.

It writes all new artifacts under:

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

## Five-Domain Score-Gap-Only Evaluation

`job4_five_domain_score_gap_eval.py` evaluates only:

- fixed lambda;
- runtime score-gap.

It runs on:

- SciFact;
- FiQA;
- NFCorpus;
- ArguAna;
- TREC-COVID.

It does not train or evaluate MLP policies.

Required restored inputs:

```bash
outputs/fineweb_labeled.json
outputs/marco_labeled.json
outputs/scifact_sweep_records.json
outputs/fiqa_sweep_records.json
outputs/nfcorpus_sweep_records.json
outputs/arguana_sweep_records.json
outputs/trec-covid_sweep_records.json
outputs/beir_data/scifact
outputs/beir_data/fiqa
outputs/beir_data/nfcorpus
outputs/beir_data/arguana
outputs/beir_data/trec-covid
```

Default output directory:

```bash
outputs/runtime_ablation/five_domain_score_gap
```

Key outputs:

```bash
outputs/runtime_ablation/five_domain_score_gap/five_domain_score_gap_runs.json
outputs/runtime_ablation/five_domain_score_gap/five_domain_score_gap_query_metrics.jsonl
outputs/runtime_ablation/five_domain_score_gap/five_domain_score_gap_bootstrap_summary.json
outputs/runtime_ablation/five_domain_score_gap/five_domain_score_gap_paper_table.csv
outputs/runtime_ablation/five_domain_score_gap/five_domain_score_gap_paper_table.tex
outputs/runtime_ablation/five_domain_score_gap/five_domain_score_gap_metadata.json
```

## Learned Runtime Feature Evaluation

`job5_learned_runtime_feature_eval.py` trains a small ridge regressor on
candidate-side score features from each query's preliminary top-50 list.

Features include:

- top scores;
- adjacent score gaps;
- score mean, standard deviation, min, max, median, and percentiles;
- entropy of normalized scores;
- score-decay slope;
- max adjacent gap and its position;
- counts above fixed score thresholds.

The model predicts `log(lambda)` and evaluates by selecting the nearest saved
lambda sweep row. By default, runtime model training is leave-domain-out: for
each held-out BEIR domain, the runtime regressor is trained on the other four
domains.

Compared methods:

- fixed lambda;
- covariance MLP;
- score-gap heuristic;
- learned runtime feature ridge model;
- nDCG oracle;
- precision oracle.

Default output directory:

```bash
outputs/runtime_ablation/learned_runtime_feature
```

Key outputs:

```bash
outputs/runtime_ablation/learned_runtime_feature/learned_runtime_feature_runs.json
outputs/runtime_ablation/learned_runtime_feature/learned_runtime_feature_query_metrics.jsonl
outputs/runtime_ablation/learned_runtime_feature/learned_runtime_feature_bootstrap_summary.json
outputs/runtime_ablation/learned_runtime_feature/learned_runtime_feature_paper_table.csv
outputs/runtime_ablation/learned_runtime_feature/learned_runtime_feature_paper_table.tex
outputs/runtime_ablation/learned_runtime_feature/learned_runtime_feature_metadata.json
outputs/runtime_ablation/learned_runtime_feature/learned_runtime_ndcg_by_domain.png
outputs/runtime_ablation/learned_runtime_feature/learned_runtime_precision_by_domain.png
outputs/runtime_ablation/learned_runtime_feature/learned_runtime_avg_returned_by_domain.png
```

## Final Consistent Evaluation Tables

`job6_final_consistent_eval.py` regenerates corrected paper tables without
running Job 1, Job 2, sweep construction, or model training.

It standardizes oracle selection:

- nDCG oracle breaks ties toward fewer admitted segments;
- precision oracle breaks ties by precision, nDCG, then fewer admitted segments;
- score-gap nDCG uses the shared `beir_ndcg(...)` scorer.

It consumes saved sweep records and, when available, existing policy-query
metric files for Plain/Hybrid/Covariance MLP chosen lambdas.

Default output directory:

```bash
outputs/runtime_ablation/final_consistent_eval
```

Key outputs:

```bash
outputs/runtime_ablation/final_consistent_eval/final_consistent_runs.json
outputs/runtime_ablation/final_consistent_eval/final_consistent_query_metrics.jsonl
outputs/runtime_ablation/final_consistent_eval/final_consistent_bootstrap_summary.json
outputs/runtime_ablation/final_consistent_eval/final_consistent_paper_table.csv
outputs/runtime_ablation/final_consistent_eval/final_consistent_paper_table.tex
outputs/runtime_ablation/final_consistent_eval/sanity_checks.json
outputs/runtime_ablation/final_consistent_eval/metadata.json
```

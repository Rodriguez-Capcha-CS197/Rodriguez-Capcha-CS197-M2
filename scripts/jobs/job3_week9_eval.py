"""Job 3: Week 9 cross-domain robustness evaluation."""

from __future__ import annotations

import argparse
import json
import logging
import os
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from beir.datasets.data_loader import GenericDataLoader

from configs.metadata import save_run_metadata
from scripts.reporting import (
    export_bootstrap_latex,
    export_domain_latex,
    export_domain_precision_latex,
    plot_feature_ablation,
    plot_lambda_distribution_shift,
    plot_tsne_lambda,
)
from shared.beir_data import resolve_beir_dataset_path
from shared.beir_scoring import build_beir_segments
from shared.dataset_utils import merge_and_shuffle_datasets
from shared.embedding import MiniLMEmbedder
from shared.lambda_inference import (
    predict_covariance_lambda,
    predict_hybrid_lambda,
    predict_plain_lambda,
)
from shared.logging_utils import configure_logging
from shared.predictor import LambdaPredictor
from shared.segments import SegmentBuildConfig


LOGGER = logging.getLogger(__name__)


METHOD_ORDER = ["oracle", "precision_oracle", "f1_oracle", "fixed", "plain", "hybrid", "covariance"]
METRIC_KEYS = [
    "ndcg_at_10",
    "precision_returned",
    "precision_at_5",
    "recall_returned",
    "recall_at_5",
    "f1_returned",
    "num_returned_docs",
]


def _train_model(X, y, seed, hidden_dim=64, epochs=200, lr=1e-3):
    torch.manual_seed(seed)
    np.random.seed(seed)
    idx = np.arange(len(X))
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)
    split = max(1, int(0.8 * len(idx)))
    train_idx = idx[:split]
    val_idx = idx[split:]
    if len(val_idx) == 0:
        val_idx = train_idx[:1]

    model = LambdaPredictor(input_dim=X.shape[1], hidden_dim=hidden_dim)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    X_train = torch.tensor(X[train_idx], dtype=torch.float32)
    y_train = torch.tensor(np.log(np.clip(y[train_idx], 1e-10, None)), dtype=torch.float32)
    X_val = torch.tensor(X[val_idx], dtype=torch.float32)
    y_val = torch.tensor(np.log(np.clip(y[val_idx], 1e-10, None)), dtype=torch.float32)

    best_state = None
    best_val = float("inf")
    for _ in range(epochs):
        model.train()
        optimizer.zero_grad()
        pred = model(X_train)
        loss = criterion(pred, y_train)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val = criterion(model(X_val), y_val).item()
        if val < best_val:
            best_val = val
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model


def _group_by_query(records):
    by_q = defaultdict(list)
    for row in records:
        by_q[row["question_id"]].append(row)
    return by_q


def _dedupe_doc_ids(doc_ids):
    return list(dict.fromkeys(str(doc_id) for doc_id in doc_ids))


def _row_retrieval_metrics(row):
    retrieved_doc_ids = _dedupe_doc_ids(row.get("retrieved_doc_ids", []))
    relevant_doc_ids = {str(doc_id) for doc_id in row.get("relevant_doc_ids", [])}
    retrieved_at_5 = retrieved_doc_ids[:5]
    hit_count_returned = sum(1 for doc_id in retrieved_doc_ids if doc_id in relevant_doc_ids)
    hit_count_at_5 = sum(1 for doc_id in retrieved_at_5 if doc_id in relevant_doc_ids)
    precision_returned = hit_count_returned / len(retrieved_doc_ids) if retrieved_doc_ids else 0.0
    recall_returned = hit_count_returned / len(relevant_doc_ids) if relevant_doc_ids else 0.0
    recall_at_5 = hit_count_at_5 / len(relevant_doc_ids) if relevant_doc_ids else 0.0
    if precision_returned + recall_returned == 0.0:
        f1_returned = 0.0
    else:
        f1_returned = 2.0 * precision_returned * recall_returned / (precision_returned + recall_returned)
    return {
        "ndcg_at_10": float(row["ndcg_at_10"]),
        "precision_returned": float(precision_returned),
        "precision_at_5": float(hit_count_at_5 / 5.0),
        "recall_returned": float(recall_returned),
        "recall_at_5": float(recall_at_5),
        "f1_returned": float(f1_returned),
        "num_returned_docs": float(len(retrieved_doc_ids)),
        "hit_count": int(hit_count_returned),
        "hit_count_at_5": int(hit_count_at_5),
        "retrieved_doc_ids": retrieved_doc_ids,
        "relevant_doc_ids": sorted(relevant_doc_ids),
    }


def _row_score(row, metric_key):
    return _row_retrieval_metrics(row)[metric_key]


def _evaluate_policy(records_by_query, policy_fn):
    query_rows = []
    for qid, rows in records_by_query.items():
        query = rows[0]["query"]
        chosen_lambda = policy_fn(qid, query, rows)
        if isinstance(chosen_lambda, dict):
            best_row = chosen_lambda
        else:
            best_row = min(rows, key=lambda r: abs(float(r["lambda"]) - float(chosen_lambda)))
        metrics = _row_retrieval_metrics(best_row)
        query_rows.append(
            {
                "question_id": qid,
                "query": query,
                "chosen_lambda": float(best_row["lambda"]),
                **metrics,
            }
        )

    aggregates = {}
    for metric_key in METRIC_KEYS:
        values = [float(row[metric_key]) for row in query_rows]
        aggregates[metric_key] = float(np.mean(values)) if values else 0.0
    return aggregates, query_rows


def _oracle_policy(_qid, _query, rows):
    best = max(rows, key=lambda r: float(r["ndcg_at_10"]))
    return best


def _precision_oracle_policy(_qid, _query, rows):
    return max(
        rows,
        key=lambda r: (
            _row_score(r, "precision_returned"),
            _row_score(r, "precision_at_5"),
            float(r["ndcg_at_10"]),
        ),
    )


def _f1_oracle_policy(_qid, _query, rows):
    return max(
        rows,
        key=lambda r: (
            _row_score(r, "f1_returned"),
            float(r["ndcg_at_10"]),
            -_row_score(r, "num_returned_docs"),
        ),
    )


def _fixed_policy(lam):
    def fn(_qid, _query, _rows):
        return lam
    return fn


def _collect_training_features(records):
    optimal = [r for r in records if r.get("is_optimal")]
    missing = [
        r["question_id"]
        for r in optimal
        if "plain_features" not in r or "hybrid_features" not in r or "covariance_features" not in r
    ]
    if missing:
        preview = ", ".join(missing[:5])
        raise ValueError(
            "Training records are missing saved feature vectors on optimal rows. "
            "Regenerate Job 2 labels with the current feature-serialization code. "
            f"Example missing qids: {preview}"
        )

    X_plain = np.asarray([r["plain_features"] for r in optimal], dtype=np.float32)
    X_hybrid = np.asarray([r["hybrid_features"] for r in optimal], dtype=np.float32)
    X_cov = np.asarray([r["covariance_features"] for r in optimal], dtype=np.float32)
    lambdas = np.asarray([float(r["lambda"]) for r in optimal], dtype=np.float32)
    return X_plain, X_hybrid, X_cov, lambdas


def _assert_record_segment_config(records, expected_config, records_path):
    strategies = {r.get("segment_strategy") for r in records if r.get("segment_strategy")}
    expected_strategy = expected_config.strategy
    if not strategies:
        print(f"warning: {records_path} has no segment_strategy metadata; assuming {expected_strategy}")
        return
    if strategies != {expected_strategy}:
        raise ValueError(
            f"{records_path} was generated with segment strategies {sorted(strategies)}, "
            f"but this run requested {expected_strategy!r}."
        )
    configs = [r.get("segment_config") for r in records if r.get("segment_config")]
    if not configs:
        return
    expected = expected_config.to_dict()
    mismatches = []
    for key, expected_value in expected.items():
        seen = {cfg.get(key) for cfg in configs}
        if seen != {expected_value}:
            mismatches.append(f"{key}: saw {sorted(seen)!r}, expected {expected_value!r}")
    if mismatches:
        raise ValueError(f"{records_path} segment_config mismatch: " + "; ".join(mismatches))


def _doc_preview(corpus, doc_id, max_chars=360):
    doc = corpus.get(str(doc_id), {})
    title = str(doc.get("title", "")).strip()
    text = str(doc.get("text", "")).strip().replace("\n", " ")
    return {
        "doc_id": str(doc_id),
        "title": title,
        "text": text[:max_chars],
    }


def _eval_domain(
    domain_name,
    records_path,
    beir_data_path,
    split,
    embedder,
    plain_model,
    hybrid_model,
    cov_model,
    fixed_lambda,
    segment_config,
    seed,
):
    with open(records_path, "r", encoding="utf-8") as f:
        records = json.load(f)
    _assert_record_segment_config(records, segment_config, records_path)
    records_by_query = _group_by_query(records)

    resolved_data_path = resolve_beir_dataset_path(beir_data_path, domain_name.lower(), split)
    corpus, _, _ = GenericDataLoader(resolved_data_path).load(split=split)
    segments, _ = build_beir_segments(corpus, embedder, segment_config=segment_config)
    corpus_embs = np.asarray([seg.vector for seg in segments], dtype=np.float32)
    corpus_norms = np.linalg.norm(corpus_embs, axis=1)

    def plain_policy(_qid, query, _rows):
        return predict_plain_lambda(plain_model, query, embedder.embed_single)

    def hybrid_policy(_qid, query, _rows):
        return predict_hybrid_lambda(hybrid_model, query, embedder.embed_single, corpus_embs, corpus_norms)

    def cov_policy(_qid, query, _rows):
        return predict_covariance_lambda(cov_model, query, embedder.embed_single, corpus_embs, corpus_norms)

    policy_results = {
        "oracle": _evaluate_policy(records_by_query, _oracle_policy),
        "precision_oracle": _evaluate_policy(records_by_query, _precision_oracle_policy),
        "f1_oracle": _evaluate_policy(records_by_query, _f1_oracle_policy),
        "fixed": _evaluate_policy(records_by_query, _fixed_policy(fixed_lambda)),
        "plain": _evaluate_policy(records_by_query, plain_policy),
        "hybrid": _evaluate_policy(records_by_query, hybrid_policy),
        "covariance": _evaluate_policy(records_by_query, cov_policy),
    }
    row = {"domain": domain_name}
    query_metric_rows = []
    for method, (metrics, query_rows) in policy_results.items():
        row[method] = metrics["ndcg_at_10"]
        for metric_key in METRIC_KEYS:
            row[f"{method}_{metric_key}"] = metrics[metric_key]
        row[f"{method}_avg_num_returned_docs"] = metrics["num_returned_docs"]
        for query_row in query_rows:
            query_metric_rows.append(
                {
                    "domain": domain_name,
                    "seed": int(seed),
                    "method": method,
                    **query_row,
                }
            )

    fixed_by_qid = {item["question_id"]: item for item in query_metric_rows if item["method"] == "fixed"}
    oracle_by_qid = {item["question_id"]: item for item in query_metric_rows if item["method"] == "oracle"}
    qualitative_rows = []
    for item in query_metric_rows:
        if item["method"] != "covariance":
            continue
        fixed = fixed_by_qid.get(item["question_id"])
        oracle = oracle_by_qid.get(item["question_id"])
        if fixed is None or oracle is None:
            continue
        precision_drop = float(fixed["precision_returned"]) - float(item["precision_returned"])
        if precision_drop <= 0.0:
            continue
        direction = "lower_lambda" if item["chosen_lambda"] < fixed["chosen_lambda"] else "higher_lambda"
        qualitative_rows.append(
            {
                "domain": domain_name,
                "seed": int(seed),
                "question_id": item["question_id"],
                "query": item["query"],
                "direction": direction,
                "precision_drop_vs_fixed": precision_drop,
                "fixed": {
                    key: fixed[key]
                    for key in [
                        "chosen_lambda",
                        "ndcg_at_10",
                        "precision_returned",
                        "precision_at_5",
                        "recall_at_5",
                        "f1_returned",
                        "num_returned_docs",
                    ]
                },
                "covariance": {
                    key: item[key]
                    for key in [
                        "chosen_lambda",
                        "ndcg_at_10",
                        "precision_returned",
                        "precision_at_5",
                        "recall_at_5",
                        "f1_returned",
                        "num_returned_docs",
                    ]
                },
                "oracle": {
                    key: oracle[key]
                    for key in [
                        "chosen_lambda",
                        "ndcg_at_10",
                        "precision_returned",
                        "precision_at_5",
                        "recall_at_5",
                        "f1_returned",
                        "num_returned_docs",
                    ]
                },
                "relevant_docs": [_doc_preview(corpus, doc_id) for doc_id in item["relevant_doc_ids"][:3]],
                "fixed_retrieved_docs": [_doc_preview(corpus, doc_id) for doc_id in fixed["retrieved_doc_ids"][:5]],
                "covariance_retrieved_docs": [_doc_preview(corpus, doc_id) for doc_id in item["retrieved_doc_ids"][:5]],
                "oracle_retrieved_docs": [_doc_preview(corpus, doc_id) for doc_id in oracle["retrieved_doc_ids"][:5]],
            }
        )
    return row, query_metric_rows, qualitative_rows


def _plot_summary(scifact_runs, fiqa_runs, out_png, out_pdf):
    import matplotlib.pyplot as plt

    labels = ["oracle", "fixed", "plain", "hybrid", "covariance"]
    x = np.arange(len(labels))
    width = 0.35

    def _mean_std(runs, key):
        vals = np.asarray([r[key] for r in runs], dtype=np.float32)
        return float(np.mean(vals)), float(np.std(vals))

    sf_mean = [_mean_std(scifact_runs, k)[0] for k in labels]
    sf_std = [_mean_std(scifact_runs, k)[1] for k in labels]
    fq_mean = [_mean_std(fiqa_runs, k)[0] for k in labels]
    fq_std = [_mean_std(fiqa_runs, k)[1] for k in labels]

    plt.figure(figsize=(10, 5))
    plt.bar(x - width / 2, sf_mean, width=width, yerr=sf_std, capsize=4, label="SciFact")
    plt.bar(x + width / 2, fq_mean, width=width, yerr=fq_std, capsize=4, label="FiQA")
    plt.xticks(x, labels, rotation=20, ha="right")
    plt.ylabel("nDCG@10")
    plt.title("Week 9 Cross-Domain Robustness")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    plt.savefig(out_pdf)


def _safe_domain_filename(domain_name):
    return domain_name.replace("/", "_").replace(" ", "_")


def _parse_domains(args):
    domains = []
    for raw_name in [item.strip() for item in args.domains.split(",") if item.strip()]:
        parts = raw_name.split(":")
        name = parts[0]
        records_path = parts[1] if len(parts) > 1 and parts[1] else f"outputs/{name}_sweep_records.json"
        data_path = parts[2] if len(parts) > 2 and parts[2] else f"outputs/beir_data/{name}"
        split = parts[3] if len(parts) > 3 and parts[3] else args.split
        if name == "scifact" and len(parts) == 1:
            records_path = args.scifact_records
            data_path = args.scifact_data_path
        elif name == "fiqa" and len(parts) == 1:
            records_path = args.fiqa_records
            data_path = args.fiqa_data_path
        domains.append(
            {
                "name": name,
                "display_name": name.replace("-", " ").title(),
                "records_path": records_path,
                "data_path": data_path,
                "split": split,
                "results_path": os.path.join(args.output_dir, f"{_safe_domain_filename(name)}_results.json"),
            }
        )
    return domains


def _bootstrap_mean_ci(values, n_samples, seed):
    values = np.asarray(values, dtype=np.float32)
    if len(values) == 0:
        return {"mean": 0.0, "ci_low": 0.0, "ci_high": 0.0}
    if n_samples <= 0 or len(values) == 1:
        mean = float(np.mean(values))
        return {"mean": mean, "ci_low": mean, "ci_high": mean}
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(n_samples, len(values)))
    means = np.mean(values[indices], axis=1)
    return {
        "mean": float(np.mean(values)),
        "ci_low": float(np.quantile(means, 0.025)),
        "ci_high": float(np.quantile(means, 0.975)),
    }


def _paired_bootstrap_diff_ci(values_a, values_b, n_samples, seed):
    values_a = np.asarray(values_a, dtype=np.float32)
    values_b = np.asarray(values_b, dtype=np.float32)
    diffs = values_a - values_b
    if len(diffs) == 0:
        return {"mean_diff": 0.0, "ci_low": 0.0, "ci_high": 0.0, "p_two_sided": 1.0}
    if n_samples <= 0 or len(diffs) == 1:
        mean = float(np.mean(diffs))
        return {"mean_diff": mean, "ci_low": mean, "ci_high": mean, "p_two_sided": 1.0}
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(diffs), size=(n_samples, len(diffs)))
    means = np.mean(diffs[indices], axis=1)
    p_two_sided = 2.0 * min(float(np.mean(means <= 0.0)), float(np.mean(means >= 0.0)))
    return {
        "mean_diff": float(np.mean(diffs)),
        "ci_low": float(np.quantile(means, 0.025)),
        "ci_high": float(np.quantile(means, 0.975)),
        "p_two_sided": min(1.0, p_two_sided),
    }


def _write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def _write_bootstrap_summary(query_metric_rows, output_path, n_samples, seed):
    summary = []
    grouped = defaultdict(lambda: defaultdict(list))
    for row in query_metric_rows:
        grouped[(row["domain"], row["method"])][row["question_id"]].append(row)

    for (domain, method), rows_by_qid in sorted(grouped.items()):
        item = {
            "domain": domain,
            "method": method,
            "n_queries": len(rows_by_qid),
            "n_seed_query_rows": sum(len(rows) for rows in rows_by_qid.values()),
        }
        for offset, metric_key in enumerate(METRIC_KEYS):
            values = [
                float(np.mean([float(row[metric_key]) for row in rows]))
                for rows in rows_by_qid.values()
            ]
            item[metric_key] = _bootstrap_mean_ci(values, n_samples, seed + offset)
            if method != "fixed":
                paired = []
                fixed = []
                fixed_by_qid = grouped.get((domain, "fixed"), {})
                for qid, rows in rows_by_qid.items():
                    fixed_rows = fixed_by_qid.get(qid)
                    if not fixed_rows:
                        continue
                    paired.append(float(np.mean([float(row[metric_key]) for row in rows])))
                    fixed.append(float(np.mean([float(row[metric_key]) for row in fixed_rows])))
                item[f"{metric_key}_diff_vs_fixed"] = _paired_bootstrap_diff_ci(
                    paired,
                    fixed,
                    n_samples,
                    seed + 100 + offset,
                )
        summary.append(item)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)


def _write_qualitative_errors(path_json, path_md, qualitative_rows, examples_per_domain):
    selected = []
    grouped = defaultdict(list)
    for row in qualitative_rows:
        grouped[(row["domain"], row["direction"])].append(row)
    for key, rows in grouped.items():
        rows = sorted(rows, key=lambda row: row["precision_drop_vs_fixed"], reverse=True)
        selected.extend(rows[:examples_per_domain])

    with open(path_json, "w", encoding="utf-8") as handle:
        json.dump(selected, handle, indent=2)

    with open(path_md, "w", encoding="utf-8") as handle:
        handle.write("# Qualitative Covariance Precision Drops\n\n")
        for row in selected:
            handle.write(f"## {row['domain']} / {row['direction']} / {row['question_id']}\n\n")
            handle.write(f"Query: {row['query']}\n\n")
            handle.write(f"Precision drop vs fixed: {row['precision_drop_vs_fixed']:.3f}\n\n")
            for label in ["fixed", "covariance", "oracle"]:
                metrics = row[label]
                handle.write(
                    f"- {label}: lambda={metrics['chosen_lambda']:.4g}, "
                    f"nDCG={metrics['ndcg_at_10']:.3f}, "
                    f"precision@returned={metrics['precision_returned']:.3f}, "
                    f"P@5={metrics['precision_at_5']:.3f}, "
                    f"recall@5={metrics['recall_at_5']:.3f}, "
                    f"returned={metrics['num_returned_docs']:.0f}\n"
                )
            handle.write("\nRelevant docs:\n")
            for doc in row["relevant_docs"]:
                handle.write(f"- {doc['doc_id']}: {doc['title']} {doc['text']}\n")
            handle.write("\nCovariance retrieved docs:\n")
            for doc in row["covariance_retrieved_docs"]:
                handle.write(f"- {doc['doc_id']}: {doc['title']} {doc['text']}\n")
            handle.write("\n")


def main():
    configure_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument("--fineweb-records", default="outputs/fineweb_labeled.json")
    parser.add_argument("--marco-records", default="outputs/marco_labeled.json")
    parser.add_argument("--scifact-records", default="outputs/scifact_sweep_records.json")
    parser.add_argument("--fiqa-records", default="outputs/fiqa_sweep_records.json")
    parser.add_argument("--scifact-data-path", default="outputs/beir_data/scifact")
    parser.add_argument("--fiqa-data-path", default="outputs/beir_data/fiqa")
    parser.add_argument(
        "--domains",
        default="scifact,fiqa",
        help=(
            "Comma-separated domains to evaluate. Use name or name:records_path:data_path:split. "
            "Default keeps the original SciFact+FiQA run."
        ),
    )
    parser.add_argument("--split", default="test")
    parser.add_argument("--embed-model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--train-limit", type=int, default=0)
    parser.add_argument("--output-dir", default="outputs/week9")
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    parser.add_argument("--qualitative-examples-per-domain", type=int, default=3)
    parser.add_argument("--segment-strategy", default="geometry_sentence")
    parser.add_argument("--min-sentence-len", type=int, default=20)
    parser.add_argument("--min-segment-size", type=int, default=2)
    parser.add_argument("--max-segment-size", type=int, default=15)
    parser.add_argument("--lookback-k", type=int, default=50)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    domains = _parse_domains(args)
    embedder = MiniLMEmbedder(args.embed_model)
    segment_config = SegmentBuildConfig(
        strategy=args.segment_strategy,
        min_sentence_len=args.min_sentence_len,
        min_segment_size=args.min_segment_size,
        max_segment_size=args.max_segment_size,
        lookback_k=args.lookback_k,
    )

    training_records = merge_and_shuffle_datasets(args.fineweb_records, args.marco_records, seed=0)
    _assert_record_segment_config(training_records, segment_config, "training records")
    if args.train_limit and args.train_limit > 0:
        training_records = training_records[: args.train_limit]
    X_plain, X_hybrid, X_cov, train_lambdas = _collect_training_features(training_records)
    if len(train_lambdas) == 0:
        raise ValueError("No optimal rows found in training records.")
    fixed_lambda = float(np.median(train_lambdas))

    runs_by_domain = {domain["name"]: [] for domain in domains}
    all_query_metric_rows = []
    all_qualitative_rows = []
    for seed in seeds:
        plain_model = _train_model(X_plain, train_lambdas, seed=seed)
        hybrid_model = _train_model(X_hybrid, train_lambdas, seed=seed)
        cov_model = _train_model(X_cov, train_lambdas, seed=seed)

        for domain in domains:
            run_row, query_metric_rows, qualitative_rows = _eval_domain(
                domain["name"],
                domain["records_path"],
                domain["data_path"],
                domain["split"],
                embedder,
                plain_model,
                hybrid_model,
                cov_model,
                fixed_lambda,
                segment_config,
                seed,
            )
            runs_by_domain[domain["name"]].append(run_row)
            all_query_metric_rows.extend(query_metric_rows)
            if seed == seeds[0]:
                all_qualitative_rows.extend(qualitative_rows)
        LOGGER.info("completed seed %d", seed)

    domain_result_paths = []
    for domain in domains:
        results_path = domain["results_path"]
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(runs_by_domain[domain["name"]], f, indent=2)
        domain_result_paths.append((domain["display_name"], results_path))

    _write_jsonl(os.path.join(args.output_dir, "policy_query_metrics.jsonl"), all_query_metric_rows)
    bootstrap_summary_path = os.path.join(args.output_dir, "bootstrap_summary.json")
    _write_bootstrap_summary(
        all_query_metric_rows,
        bootstrap_summary_path,
        args.bootstrap_samples,
        args.bootstrap_seed,
    )
    _write_qualitative_errors(
        os.path.join(args.output_dir, "qualitative_errors.json"),
        os.path.join(args.output_dir, "qualitative_errors.md"),
        all_qualitative_rows,
        args.qualitative_examples_per_domain,
    )
    save_run_metadata(
        os.path.join(args.output_dir, "week9_metadata.json"),
        args,
        extra={
            "segment_config": segment_config.to_dict(),
            "seeds": seeds,
            "train_records": int(len(training_records)),
            "domains": domains,
        },
    )

    export_domain_latex(domain_result_paths, os.path.join(args.output_dir, "cross_domain_ndcg_table.tex"))
    export_domain_precision_latex(domain_result_paths, os.path.join(args.output_dir, "cross_domain_precision_table.tex"))
    export_bootstrap_latex(bootstrap_summary_path, os.path.join(args.output_dir, "bootstrap_summary_table.tex"))

    if "scifact" in runs_by_domain and "fiqa" in runs_by_domain:
        _plot_summary(
            runs_by_domain["scifact"],
            runs_by_domain["fiqa"],
            os.path.join(args.output_dir, "week9_comparison_table.png"),
            os.path.join(args.output_dir, "week9_comparison_table.pdf"),
        )
        plot_feature_ablation(
            runs_by_domain["scifact"],
            runs_by_domain["fiqa"],
            os.path.join(args.output_dir, "feature_ablation.png"),
        )
    try:
        plot_lambda_distribution_shift(
            args.fineweb_records,
            args.marco_records,
            os.path.join(args.output_dir, "lambda_distribution_shift.png"),
        )
    except Exception as exc:
        LOGGER.warning("lambda distribution plot skipped: %s", exc)
    try:
        plot_tsne_lambda(
            training_records,
            os.path.join(args.output_dir, "query_tsne_by_lambda.png"),
            seed=seeds[0] if seeds else 0,
        )
    except Exception as exc:
        LOGGER.warning("t-SNE plot skipped: %s", exc)

    LOGGER.info("saved week9 outputs")


if __name__ == "__main__":
    main()

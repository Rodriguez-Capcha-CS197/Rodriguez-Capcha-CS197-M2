"""Isolated runtime score-gap ablation.

This job intentionally leaves the existing Week 9 evaluator unchanged. It
reuses the same saved label/sweep records and writes every new artifact under a
separate output directory.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from beir.datasets.data_loader import GenericDataLoader

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from configs.metadata import save_run_metadata
from shared.beir_data import resolve_beir_dataset_path
from shared.beir_scoring import beir_ndcg, build_beir_segments
from shared.constants import EPS, MAX_SEGMENTS
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

METHOD_ORDER = [
    "oracle",
    "precision_oracle",
    "f1_oracle",
    "fixed",
    "plain",
    "hybrid",
    "covariance",
    "runtime_score_gap",
]

METRIC_KEYS = [
    "ndcg_at_10",
    "precision_returned",
    "precision_at_5",
    "recall_returned",
    "recall_at_5",
    "f1_returned",
    "num_returned_docs",
]


def score_gap_cutoff_indices(
    scores: np.ndarray,
    min_k: int = 1,
    max_k: int = MAX_SEGMENTS,
    candidate_k: int = 50,
) -> list[int]:
    """Return top-k indices where k is chosen by the largest adjacent score gap."""
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    if scores.size == 0:
        return []

    min_k = max(1, int(min_k))
    max_k = max(min_k, min(int(max_k), int(scores.size)))
    candidate_k = max(max_k, min(int(candidate_k), int(scores.size)))

    order = np.argsort(-scores)[:candidate_k]
    if max_k <= min_k or len(order) <= min_k:
        return [int(idx) for idx in order[:min_k]]

    sorted_scores = scores[order]
    gap_limit = min(max_k, len(sorted_scores) - 1)
    if gap_limit < min_k:
        return [int(idx) for idx in order[:min_k]]

    gaps = sorted_scores[:gap_limit] - sorted_scores[1 : gap_limit + 1]
    k_star = max(min_k, min(max_k, int(np.argmax(gaps)) + 1))
    return [int(idx) for idx in order[:k_star]]


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
    y_train = torch.tensor(np.log(np.clip(y[train_idx], EPS, None)), dtype=torch.float32)
    X_val = torch.tensor(X[val_idx], dtype=torch.float32)
    y_val = torch.tensor(np.log(np.clip(y[val_idx], EPS, None)), dtype=torch.float32)

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


def _read_json_rows(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    if isinstance(data, list):
        return data
    return [data]


def _group_by_query(records: list[dict]) -> dict[str, list[dict]]:
    by_q = defaultdict(list)
    for row in records:
        by_q[str(row["question_id"])].append(row)
    return dict(by_q)


def _dedupe_doc_ids(doc_ids) -> list[str]:
    return list(dict.fromkeys(str(doc_id) for doc_id in doc_ids))


def _row_retrieval_metrics(row: dict) -> dict:
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


def _row_score(row: dict, metric_key: str) -> float:
    return float(_row_retrieval_metrics(row)[metric_key])


def _evaluate_policy(records_by_query: dict[str, list[dict]], policy_fn):
    query_rows = []
    for qid, rows in records_by_query.items():
        query = rows[0]["query"]
        choice = policy_fn(qid, query, rows)
        if isinstance(choice, dict):
            selected_row = choice
            selected_lambda = selected_row.get("lambda")
        else:
            selected_row = min(rows, key=lambda r: abs(float(r["lambda"]) - float(choice)))
            selected_lambda = float(selected_row["lambda"])

        metrics = _row_retrieval_metrics(selected_row)
        query_rows.append(
            {
                "question_id": qid,
                "query": query,
                "chosen_lambda": None if selected_lambda is None else float(selected_lambda),
                **metrics,
            }
        )

    aggregates = {}
    for metric_key in METRIC_KEYS:
        values = [float(row[metric_key]) for row in query_rows]
        aggregates[metric_key] = float(np.mean(values)) if values else 0.0
    return aggregates, query_rows


def _oracle_policy(_qid, _query, rows):
    # nDCG ties are broken toward fewer admitted segments to avoid treating
    # equally ranked but more permissive retrieval as oracle-superior.
    return max(rows, key=lambda row: (float(row["ndcg_at_10"]), -int(row.get("num_segments", 0))))


def _precision_oracle_policy(_qid, _query, rows):
    return max(
        rows,
        key=lambda r: (
            _row_score(r, "precision_returned"),
            float(r["ndcg_at_10"]),
            -int(r.get("num_segments", _row_score(r, "num_returned_docs"))),
        ),
    )


def _f1_oracle_policy(_qid, _query, rows):
    return max(
        rows,
        key=lambda r: (
            _row_score(r, "f1_returned"),
            float(r["ndcg_at_10"]),
            -int(r.get("num_segments", _row_score(r, "num_returned_docs"))),
        ),
    )


def _fixed_policy(lam):
    def fn(_qid, _query, _rows):
        return lam

    return fn


def _collect_training_features(records: list[dict]):
    optimal = [row for row in records if row.get("is_optimal")]
    missing = [
        row["question_id"]
        for row in optimal
        if "plain_features" not in row or "hybrid_features" not in row or "covariance_features" not in row
    ]
    if missing:
        preview = ", ".join(str(qid) for qid in missing[:5])
        raise ValueError(
            "Training records are missing saved feature vectors on optimal rows. "
            "Regenerate or restore Job 2 labels with feature serialization. "
            f"Example missing qids: {preview}"
        )

    X_plain = np.asarray([row["plain_features"] for row in optimal], dtype=np.float32)
    X_hybrid = np.asarray([row["hybrid_features"] for row in optimal], dtype=np.float32)
    X_cov = np.asarray([row["covariance_features"] for row in optimal], dtype=np.float32)
    lambdas = np.asarray([float(row["lambda"]) for row in optimal], dtype=np.float32)
    return X_plain, X_hybrid, X_cov, lambdas


def _assert_record_segment_config(records: list[dict], expected_config: SegmentBuildConfig, records_path: str) -> None:
    strategies = {row.get("segment_strategy") for row in records if row.get("segment_strategy")}
    if strategies and strategies != {expected_config.strategy}:
        raise ValueError(
            f"{records_path} was generated with segment strategies {sorted(strategies)}, "
            f"but this run requested {expected_config.strategy!r}."
        )

    configs = [row.get("segment_config") for row in records if row.get("segment_config")]
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


def _runtime_score_gap_row(
    qid,
    query,
    rows,
    segments,
    segment_to_doc_id,
    qrels,
    embedder,
    corpus_embs,
    corpus_norms,
    candidate_k,
    max_segments,
) -> dict:
    query_emb = np.asarray(embedder.embed_single(query), dtype=np.float32).reshape(-1)
    query_norm = float(np.linalg.norm(query_emb))
    denom = np.maximum(corpus_norms * max(query_norm, EPS), EPS)
    scores = (corpus_embs @ query_emb) / denom
    selected_indices = score_gap_cutoff_indices(
        scores,
        min_k=1,
        max_k=max_segments,
        candidate_k=candidate_k,
    )
    selected_segments = [segments[idx] for idx in selected_indices]
    retrieved_doc_ids = [
        segment_to_doc_id[int(segment.start_idx)]
        for segment in selected_segments
        if int(segment.start_idx) in segment_to_doc_id
    ]
    return {
        "question_id": qid,
        "query": query,
        "lambda": None,
        "ndcg_at_10": beir_ndcg(
            retrieved_segments=selected_segments,
            query_id=qid,
            qrels=qrels,
            segment_to_doc_id=segment_to_doc_id,
            k=10,
        ),
        "num_segments": int(len(selected_segments)),
        "relevant_doc_ids": list(rows[0].get("relevant_doc_ids", [])),
        "retrieved_doc_ids": retrieved_doc_ids,
        "runtime_score_gap_k": int(len(selected_segments)),
        "runtime_candidate_k": int(min(candidate_k, len(scores))),
    }


def _eval_domain(
    domain_name: str,
    records_path: str,
    beir_data_path: str,
    split: str,
    embedder: MiniLMEmbedder,
    plain_model,
    hybrid_model,
    cov_model,
    fixed_lambda: float,
    segment_config: SegmentBuildConfig,
    seed: int,
    candidate_k: int,
    max_segments: int,
):
    records = _read_json_rows(records_path)
    _assert_record_segment_config(records, segment_config, records_path)
    records_by_query = _group_by_query(records)

    resolved_data_path = resolve_beir_dataset_path(beir_data_path, domain_name.lower(), split)
    corpus, _, qrels = GenericDataLoader(resolved_data_path).load(split=split)
    segments, segment_to_doc_id = build_beir_segments(corpus, embedder, segment_config=segment_config)
    corpus_embs = np.asarray([segment.vector for segment in segments], dtype=np.float32)
    corpus_norms = np.linalg.norm(corpus_embs, axis=1)

    def plain_policy(_qid, query, _rows):
        return predict_plain_lambda(plain_model, query, embedder.embed_single)

    def hybrid_policy(_qid, query, _rows):
        return predict_hybrid_lambda(hybrid_model, query, embedder.embed_single, corpus_embs, corpus_norms)

    def cov_policy(_qid, query, _rows):
        return predict_covariance_lambda(cov_model, query, embedder.embed_single, corpus_embs, corpus_norms)

    def runtime_score_gap_policy(qid, query, rows):
        return _runtime_score_gap_row(
            qid,
            query,
            rows,
            segments,
            segment_to_doc_id,
            qrels,
            embedder,
            corpus_embs,
            corpus_norms,
            candidate_k,
            max_segments,
        )

    policy_results = {
        "oracle": _evaluate_policy(records_by_query, _oracle_policy),
        "precision_oracle": _evaluate_policy(records_by_query, _precision_oracle_policy),
        "f1_oracle": _evaluate_policy(records_by_query, _f1_oracle_policy),
        "fixed": _evaluate_policy(records_by_query, _fixed_policy(fixed_lambda)),
        "plain": _evaluate_policy(records_by_query, plain_policy),
        "hybrid": _evaluate_policy(records_by_query, hybrid_policy),
        "covariance": _evaluate_policy(records_by_query, cov_policy),
        "runtime_score_gap": _evaluate_policy(records_by_query, runtime_score_gap_policy),
    }

    run_row = {
        "domain": domain_name,
        "seed": int(seed),
        "num_queries": int(len(records_by_query)),
        "num_corpus_segments": int(len(segments)),
    }
    query_metric_rows = []
    for method in METHOD_ORDER:
        metrics, query_rows = policy_results[method]
        run_row[method] = metrics["ndcg_at_10"]
        for metric_key in METRIC_KEYS:
            run_row[f"{method}_{metric_key}"] = metrics[metric_key]
        run_row[f"{method}_avg_num_returned_docs"] = metrics["num_returned_docs"]
        for query_row in query_rows:
            query_metric_rows.append(
                {
                    "domain": domain_name,
                    "seed": int(seed),
                    "method": method,
                    **query_row,
                }
            )

    return run_row, query_metric_rows


def _parse_domains(args) -> list[dict]:
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
                "results_path": os.path.join(
                    args.output_dir,
                    f"{_safe_domain_filename(name)}_runtime_score_gap_results.json",
                ),
            }
        )
    return domains


def _safe_domain_filename(domain_name: str) -> str:
    return domain_name.replace("/", "_").replace(" ", "_")


def _require_file(path: str, label: str) -> None:
    if not Path(path).is_file():
        raise FileNotFoundError(f"Missing {label}: {path}")


def _bootstrap_mean_ci(values, n_samples: int, seed: int) -> dict:
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


def _paired_bootstrap_diff_ci(values_a, values_b, n_samples: int, seed: int) -> dict:
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


def _write_jsonl(path: str, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def _write_bootstrap_summary(query_metric_rows, output_path: str, n_samples: int, seed: int) -> None:
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
        fixed_by_qid = grouped.get((domain, "fixed"), {})
        for offset, metric_key in enumerate(METRIC_KEYS):
            values = [
                float(np.mean([float(row[metric_key]) for row in rows]))
                for rows in rows_by_qid.values()
            ]
            item[metric_key] = _bootstrap_mean_ci(values, n_samples, seed + offset)
            if method != "fixed":
                paired = []
                fixed = []
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


def _mean_std(values: list[float]) -> str:
    if not values:
        return "0.000 +/- 0.000"
    arr = np.asarray(values, dtype=np.float32)
    return f"{float(np.mean(arr)):.3f} +/- {float(np.std(arr)):.3f}"


def _write_summary_tables(run_rows: list[dict], output_dir: str) -> None:
    table_rows = []
    by_domain = defaultdict(list)
    for row in run_rows:
        by_domain[row["domain"]].append(row)

    for domain, rows in sorted(by_domain.items()):
        for method in METHOD_ORDER:
            table_rows.append(
                {
                    "domain": domain,
                    "method": method,
                    "nDCG@10": _mean_std([float(row[f"{method}_ndcg_at_10"]) for row in rows]),
                    "precision@returned": _mean_std(
                        [float(row[f"{method}_precision_returned"]) for row in rows]
                    ),
                    "P@5": _mean_std([float(row[f"{method}_precision_at_5"]) for row in rows]),
                    "Recall@5": _mean_std([float(row[f"{method}_recall_at_5"]) for row in rows]),
                    "F1@returned": _mean_std([float(row[f"{method}_f1_returned"]) for row in rows]),
                    "Avg returned": _mean_std(
                        [float(row[f"{method}_avg_num_returned_docs"]) for row in rows]
                    ),
                }
            )

    df = pd.DataFrame(table_rows)
    df.to_csv(os.path.join(output_dir, "runtime_score_gap_summary.csv"), index=False)
    with open(os.path.join(output_dir, "runtime_score_gap_summary.tex"), "w", encoding="utf-8") as handle:
        handle.write(df.to_latex(index=False))


def main() -> None:
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
        help="Comma-separated domains. Use name or name:records_path:data_path:split.",
    )
    parser.add_argument("--split", default="test")
    parser.add_argument("--embed-model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--train-limit", type=int, default=0)
    parser.add_argument("--output-dir", default="outputs/runtime_score_gap_ablation")
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    parser.add_argument("--runtime-candidate-k", type=int, default=50)
    parser.add_argument("--runtime-max-segments", type=int, default=MAX_SEGMENTS)
    parser.add_argument("--segment-strategy", default="geometry_sentence")
    parser.add_argument("--min-sentence-len", type=int, default=20)
    parser.add_argument("--min-segment-size", type=int, default=2)
    parser.add_argument("--max-segment-size", type=int, default=15)
    parser.add_argument("--lookback-k", type=int, default=50)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    _require_file(args.fineweb_records, "FineWeb labeled records")
    _require_file(args.marco_records, "MS MARCO labeled records")

    domains = _parse_domains(args)
    for domain in domains:
        _require_file(domain["records_path"], f"{domain['name']} sweep records")

    seeds = [int(item) for item in args.seeds.split(",") if item.strip()]
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

    all_run_rows = []
    all_query_metric_rows = []
    for seed in seeds:
        LOGGER.info("training lambda predictors for seed %d", seed)
        plain_model = _train_model(X_plain, train_lambdas, seed=seed)
        hybrid_model = _train_model(X_hybrid, train_lambdas, seed=seed)
        cov_model = _train_model(X_cov, train_lambdas, seed=seed)

        for domain in domains:
            LOGGER.info("evaluating %s seed %d", domain["name"], seed)
            run_row, query_metric_rows = _eval_domain(
                domain_name=domain["name"],
                records_path=domain["records_path"],
                beir_data_path=domain["data_path"],
                split=domain["split"],
                embedder=embedder,
                plain_model=plain_model,
                hybrid_model=hybrid_model,
                cov_model=cov_model,
                fixed_lambda=fixed_lambda,
                segment_config=segment_config,
                seed=seed,
                candidate_k=args.runtime_candidate_k,
                max_segments=args.runtime_max_segments,
            )
            all_run_rows.append(run_row)
            all_query_metric_rows.extend(query_metric_rows)

    with open(os.path.join(args.output_dir, "runtime_score_gap_runs.json"), "w", encoding="utf-8") as handle:
        json.dump(all_run_rows, handle, indent=2)

    for domain in domains:
        domain_rows = [row for row in all_run_rows if row["domain"] == domain["name"]]
        with open(domain["results_path"], "w", encoding="utf-8") as handle:
            json.dump(domain_rows, handle, indent=2)

    _write_jsonl(
        os.path.join(args.output_dir, "runtime_score_gap_policy_query_metrics.jsonl"),
        all_query_metric_rows,
    )
    _write_summary_tables(all_run_rows, args.output_dir)
    bootstrap_path = os.path.join(args.output_dir, "runtime_score_gap_bootstrap_summary.json")
    _write_bootstrap_summary(
        all_query_metric_rows,
        bootstrap_path,
        args.bootstrap_samples,
        args.bootstrap_seed,
    )

    save_run_metadata(
        os.path.join(args.output_dir, "runtime_score_gap_metadata.json"),
        args,
        extra={
            "methods": METHOD_ORDER,
            "fixed_lambda": fixed_lambda,
            "segment_config": segment_config.to_dict(),
            "seeds": seeds,
            "train_records": int(len(training_records)),
            "runtime_candidate_k": int(args.runtime_candidate_k),
            "runtime_max_segments": int(args.runtime_max_segments),
            "domains": domains,
        },
    )
    LOGGER.info("saved runtime score-gap ablation outputs to %s", args.output_dir)


if __name__ == "__main__":
    main()

"""Job 3: Week 9 cross-domain robustness evaluation."""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from beir.datasets.data_loader import GenericDataLoader
from sentence_transformers import SentenceTransformer

from shared.beir_scoring import build_beir_segments
from shared.dataset_utils import merge_and_shuffle_datasets
from shared.lambda_inference import (
    extract_covariance_features,
    extract_density_features,
    predict_covariance_lambda,
    predict_hybrid_lambda,
    predict_plain_lambda,
)
from shared.predictor import LambdaPredictor


class MiniLMEmbedder:
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)

    def embed(self, sentences):
        return self.model.encode(sentences, convert_to_numpy=True)

    def embed_single(self, text):
        return self.embed([text])[0]


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


def _evaluate_policy(records_by_query, policy_fn):
    scores = []
    for qid, rows in records_by_query.items():
        query = rows[0]["query"]
        chosen_lambda = float(policy_fn(qid, query, rows))
        best_row = min(rows, key=lambda r: abs(float(r["lambda"]) - chosen_lambda))
        scores.append(float(best_row["ndcg_at_10"]))
    return float(np.mean(scores)) if scores else 0.0


def _oracle_policy(_qid, _query, rows):
    best = max(rows, key=lambda r: float(r["ndcg_at_10"]))
    return float(best["lambda"])


def _fixed_policy(lam):
    def fn(_qid, _query, _rows):
        return lam
    return fn


def _collect_training_targets(records):
    optimal = [r for r in records if r.get("is_optimal")]
    queries = [r["query"] for r in optimal]
    lambdas = np.asarray([float(r["lambda"]) for r in optimal], dtype=np.float32)
    return queries, lambdas


def _build_train_features(queries, embed_fn, corpus_embs, corpus_norms):
    X_plain = []
    X_hybrid = []
    X_cov = []
    for q in queries:
        q_emb = np.asarray(embed_fn(q), dtype=np.float32).reshape(-1)
        X_plain.append(q_emb)
        dens = np.asarray(extract_density_features(q_emb, corpus_embs, corpus_norms), dtype=np.float32)
        cov = np.asarray(extract_covariance_features(q_emb, corpus_embs, corpus_norms), dtype=np.float32)
        X_hybrid.append(np.concatenate([q_emb, dens], axis=0))
        X_cov.append(np.concatenate([q_emb, cov], axis=0))
    return np.asarray(X_plain), np.asarray(X_hybrid), np.asarray(X_cov)


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
):
    with open(records_path, "r", encoding="utf-8") as f:
        records = json.load(f)
    records_by_query = _group_by_query(records)

    corpus, _, _ = GenericDataLoader(beir_data_path).load(split=split)
    segments, _ = build_beir_segments(corpus, embedder)
    corpus_embs = np.asarray([seg.vector for seg in segments], dtype=np.float32)
    corpus_norms = np.linalg.norm(corpus_embs, axis=1)

    def plain_policy(_qid, query, _rows):
        return predict_plain_lambda(plain_model, query, embedder.embed_single)

    def hybrid_policy(_qid, query, _rows):
        return predict_hybrid_lambda(hybrid_model, query, embedder.embed_single, corpus_embs, corpus_norms)

    def cov_policy(_qid, query, _rows):
        return predict_covariance_lambda(cov_model, query, embedder.embed_single, corpus_embs, corpus_norms)

    return {
        "domain": domain_name,
        "oracle": _evaluate_policy(records_by_query, _oracle_policy),
        "fixed": _evaluate_policy(records_by_query, _fixed_policy(fixed_lambda)),
        "plain": _evaluate_policy(records_by_query, plain_policy),
        "hybrid": _evaluate_policy(records_by_query, hybrid_policy),
        "covariance": _evaluate_policy(records_by_query, cov_policy),
    }


def _plot_summary(scifact_runs, fiqa_runs, out_png, out_pdf):
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fineweb-records", default="outputs/fineweb_labeled.json")
    parser.add_argument("--marco-records", default="outputs/marco_labeled.json")
    parser.add_argument("--scifact-records", default="outputs/scifact_sweep_records.json")
    parser.add_argument("--fiqa-records", default="outputs/fiqa_sweep_records.json")
    parser.add_argument("--scifact-data-path", default="outputs/beir_data/scifact")
    parser.add_argument("--fiqa-data-path", default="outputs/beir_data/fiqa")
    parser.add_argument("--split", default="test")
    parser.add_argument("--embed-model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--train-limit", type=int, default=0)
    parser.add_argument("--output-dir", default="outputs/week9")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    embedder = MiniLMEmbedder(args.embed_model)

    training_records = merge_and_shuffle_datasets(args.fineweb_records, args.marco_records, seed=0)
    if args.train_limit and args.train_limit > 0:
        training_records = training_records[: args.train_limit]
    train_queries, train_lambdas = _collect_training_targets(training_records)
    if len(train_queries) == 0:
        raise ValueError("No optimal rows found in training records.")
    fixed_lambda = float(np.median(train_lambdas))

    # Build a training-corpus matrix from labeled records for feature training.
    # Hybrid/covariance test-time features are still computed on test corpus.
    train_query_embs = np.asarray([embedder.embed_single(q) for q in train_queries], dtype=np.float32)
    train_norms = np.linalg.norm(train_query_embs, axis=1)
    X_plain, X_hybrid, X_cov = _build_train_features(
        train_queries, embedder.embed_single, train_query_embs, train_norms
    )

    scifact_runs = []
    fiqa_runs = []
    for seed in seeds:
        plain_model = _train_model(X_plain, train_lambdas, seed=seed)
        hybrid_model = _train_model(X_hybrid, train_lambdas, seed=seed)
        cov_model = _train_model(X_cov, train_lambdas, seed=seed)

        scifact_runs.append(
            _eval_domain(
                "scifact",
                args.scifact_records,
                args.scifact_data_path,
                args.split,
                embedder,
                plain_model,
                hybrid_model,
                cov_model,
                fixed_lambda,
            )
        )
        fiqa_runs.append(
            _eval_domain(
                "fiqa",
                args.fiqa_records,
                args.fiqa_data_path,
                args.split,
                embedder,
                plain_model,
                hybrid_model,
                cov_model,
                fixed_lambda,
            )
        )
        print(f"completed seed {seed}")

    with open(os.path.join(args.output_dir, "scifact_results.json"), "w", encoding="utf-8") as f:
        json.dump(scifact_runs, f, indent=2)
    with open(os.path.join(args.output_dir, "fiqa_results.json"), "w", encoding="utf-8") as f:
        json.dump(fiqa_runs, f, indent=2)

    _plot_summary(
        scifact_runs,
        fiqa_runs,
        os.path.join(args.output_dir, "week9_comparison_table.png"),
        os.path.join(args.output_dir, "week9_comparison_table.pdf"),
    )

    print("saved week9 outputs")


if __name__ == "__main__":
    main()

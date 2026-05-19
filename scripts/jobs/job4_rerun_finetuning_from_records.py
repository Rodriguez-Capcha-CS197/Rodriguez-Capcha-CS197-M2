"""Rerun Job 4 k-shot fine-tuning from saved lambda sweep records.

This skips the expensive FineWeb/SciFact lambda sweeps by loading:
  - FineWeb optimal lambda labels from outputs/fineweb_labeled.json
  - SciFact per-lambda sweep rows from outputs/scifact_sweep_records.json

Predicted lambdas are evaluated by snapping to the nearest lambda value present
in the saved SciFact sweep records.
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import os
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from configs.metadata import save_run_metadata
from scripts.reporting import export_kshot_latex
from shared.embedding import MiniLMEmbedder
from shared.io_utils import load_records
from shared.logging_utils import configure_logging
from shared.predictor import LambdaPredictor
from shared.scoring import ensure_1d


LOGGER = logging.getLogger(__name__)


def _optimal_rows(records: list[dict]) -> list[dict]:
    return [row for row in records if row.get("is_optimal")]


def _embed_queries(rows: list[dict], embedder: MiniLMEmbedder) -> np.ndarray:
    embeddings = []
    for i, row in enumerate(rows, start=1):
        embeddings.append(ensure_1d(embedder.embed_single(str(row["query"]))).astype(np.float32))
        if i % 1000 == 0:
            LOGGER.info("embedded %d/%d training queries", i, len(rows))
    return np.asarray(embeddings, dtype=np.float32)


def _train_predictor(
    X: np.ndarray,
    y: np.ndarray,
    seed: int,
    hidden_dim: int,
    epochs: int,
    lr: float,
) -> LambdaPredictor:
    torch.manual_seed(seed)
    np.random.seed(seed)

    rng = np.random.default_rng(seed)
    idx = np.arange(len(X))
    rng.shuffle(idx)
    split = max(1, int(0.8 * len(idx)))
    train_idx = idx[:split]
    val_idx = idx[split:] if split < len(idx) else idx[:1]

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
            val_loss = criterion(model(X_val), y_val).item()
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model


def _few_shot_finetune(
    base_model: LambdaPredictor,
    X_shot: np.ndarray,
    y_shot: np.ndarray,
    epochs: int,
    lr: float,
) -> LambdaPredictor:
    model = copy.deepcopy(base_model)
    if len(X_shot) == 0:
        model.eval()
        return model

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    X = torch.tensor(X_shot, dtype=torch.float32)
    y = torch.tensor(np.log(np.clip(y_shot, 1e-10, None)), dtype=torch.float32)

    for _ in range(epochs):
        model.train()
        optimizer.zero_grad()
        pred = model(X)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()
    model.eval()
    return model


def _predict_lambda(model: LambdaPredictor, q_emb: np.ndarray) -> float:
    with torch.no_grad():
        log_lam = model(torch.tensor(q_emb, dtype=torch.float32).unsqueeze(0)).item()
    return float(np.exp(log_lam))


def _group_scifact_records(records: list[dict]) -> dict[str, list[dict]]:
    grouped = defaultdict(list)
    for row in records:
        grouped[str(row["question_id"])].append(row)
    for rows in grouped.values():
        rows.sort(key=lambda row: float(row["lambda"]))
    return dict(grouped)


def _nearest_row(rows: list[dict], lam: float) -> dict:
    return min(rows, key=lambda row: abs(float(row["lambda"]) - float(lam)))


def _row_ndcg(row: dict) -> float:
    return float(row.get("ndcg_at_10", 0.0))


def _run_kshot_from_sweeps(
    scifact_records: list[dict],
    embedder: MiniLMEmbedder,
    train_model: LambdaPredictor,
    fixed_lambda: float,
    k_shot: int,
    seed: int,
    ft_epochs: int,
    ft_lr: float,
) -> dict:
    rows_by_qid = _group_scifact_records(scifact_records)
    optimal_by_qid = {}
    query_by_qid = {}
    for qid, rows in rows_by_qid.items():
        optimal = [row for row in rows if row.get("is_optimal")]
        optimal_by_qid[qid] = optimal[0] if optimal else max(rows, key=_row_ndcg)
        query_by_qid[qid] = str(rows[0]["query"])

    qids = sorted(rows_by_qid)
    rng = np.random.default_rng(seed)
    shuffled = list(qids)
    rng.shuffle(shuffled)
    shot_qids = set(shuffled[: min(k_shot, len(shuffled))])
    held_out_qids = [qid for qid in qids if qid not in shot_qids]

    scifact_embeddings = {
        qid: ensure_1d(embedder.embed_single(query_by_qid[qid])).astype(np.float32)
        for qid in qids
    }
    shot_X = np.asarray([scifact_embeddings[qid] for qid in qids if qid in shot_qids], dtype=np.float32)
    shot_y = np.asarray([float(optimal_by_qid[qid]["lambda"]) for qid in qids if qid in shot_qids], dtype=np.float32)

    ft_model = _few_shot_finetune(
        train_model,
        shot_X if len(shot_X) else np.empty((0, train_model.net[0].in_features), dtype=np.float32),
        shot_y,
        epochs=ft_epochs,
        lr=ft_lr,
    )

    oracle_by_qid = {}
    fixed_by_qid = {}
    zero_by_qid = {}
    few_by_qid = {}
    for qid in qids:
        rows = rows_by_qid[qid]
        q_emb = scifact_embeddings[qid]
        oracle_by_qid[qid] = _row_ndcg(optimal_by_qid[qid])
        fixed_by_qid[qid] = _row_ndcg(_nearest_row(rows, fixed_lambda))
        zero_lam = _predict_lambda(train_model, q_emb)
        zero_by_qid[qid] = _row_ndcg(_nearest_row(rows, zero_lam))
        if qid not in shot_qids:
            ft_lam = _predict_lambda(ft_model, q_emb)
            few_by_qid[qid] = _row_ndcg(_nearest_row(rows, ft_lam))

    oracle_all = float(np.mean(list(oracle_by_qid.values()))) if oracle_by_qid else 0.0
    fixed_all = float(np.mean(list(fixed_by_qid.values()))) if fixed_by_qid else 0.0
    zero_all = float(np.mean(list(zero_by_qid.values()))) if zero_by_qid else 0.0
    oracle_held = float(np.mean([oracle_by_qid[qid] for qid in held_out_qids])) if held_out_qids else 0.0
    zero_held = float(np.mean([zero_by_qid[qid] for qid in held_out_qids])) if held_out_qids else 0.0
    few_held = float(np.mean([few_by_qid[qid] for qid in held_out_qids])) if held_out_qids else 0.0
    gap = oracle_held - zero_held
    recovery_pct = float(100.0 * (few_held - zero_held) / gap) if gap > 1e-9 else 0.0

    print(f"\n=== Fine-tuning Recovery from Saved Sweeps (k={k_shot}) ===")
    print(f"Oracle nDCG@10 (all):                 {oracle_all:.3f}")
    print(f"Fixed-lambda nDCG@10 (all):           {fixed_all:.3f}")
    print(f"MLP zero-shot nDCG@10 (all):          {zero_all:.3f}")
    print(f"--- held-out queries only (n={len(held_out_qids)}) ---")
    print(f"Oracle nDCG@10:                       {oracle_held:.3f}")
    print(f"MLP zero-shot nDCG@10:                {zero_held:.3f}")
    print(f"MLP {k_shot}-shot fine-tuned nDCG@10: {few_held:.3f}")
    print(f"Oracle gap:                           {gap:.3f}")
    print(f"Recovery:                             {recovery_pct:.1f}% of oracle gap")

    return {
        "oracle_ndcg_at_10": oracle_all,
        "fixed_ndcg_at_10": fixed_all,
        "mlp_zero_shot_ndcg_at_10": zero_all,
        "mlp_zero_shot_held_out_ndcg_at_10": zero_held,
        "mlp_k_shot_ft_ndcg_at_10": few_held,
        "oracle_held_out_ndcg_at_10": oracle_held,
        "recovery_pct": recovery_pct,
        "k_shot": int(k_shot),
        "held_out_eval_queries": int(len(held_out_qids)),
        "num_scifact_queries": int(len(qids)),
    }


def main() -> None:
    configure_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument("--fineweb-records", default="outputs/fineweb_labeled.json")
    parser.add_argument("--scifact-records", default="outputs/scifact_sweep_records.json")
    parser.add_argument("--embed-model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--k-shot", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train-limit", type=int, default=0)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--train-epochs", type=int, default=200)
    parser.add_argument("--train-lr", type=float, default=1e-3)
    parser.add_argument("--ft-epochs", type=int, default=80)
    parser.add_argument("--ft-lr", type=float, default=5e-4)
    parser.add_argument("--output-path", default="outputs/job4_finetune_rerun_from_records.json")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    embedder = MiniLMEmbedder(args.embed_model)

    LOGGER.info("loading FineWeb records: %s", args.fineweb_records)
    fineweb_rows = _optimal_rows(load_records(args.fineweb_records))
    if args.train_limit and args.train_limit > 0:
        fineweb_rows = fineweb_rows[: args.train_limit]
    if not fineweb_rows:
        raise ValueError("No optimal FineWeb rows found.")

    LOGGER.info("training base predictor from %d saved FineWeb labels", len(fineweb_rows))
    X_train = _embed_queries(fineweb_rows, embedder)
    y_train = np.asarray([float(row["lambda"]) for row in fineweb_rows], dtype=np.float32)
    model = _train_predictor(
        X_train,
        y_train,
        seed=args.seed,
        hidden_dim=args.hidden_dim,
        epochs=args.train_epochs,
        lr=args.train_lr,
    )
    fixed_lambda = float(np.median(y_train))

    LOGGER.info("loading SciFact sweep records: %s", args.scifact_records)
    scifact_records = load_records(args.scifact_records)
    result = _run_kshot_from_sweeps(
        scifact_records=scifact_records,
        embedder=embedder,
        train_model=model,
        fixed_lambda=fixed_lambda,
        k_shot=args.k_shot,
        seed=args.seed,
        ft_epochs=args.ft_epochs,
        ft_lr=args.ft_lr,
    )
    lambda_grid = sorted({float(row["lambda"]) for row in scifact_records})
    result["fixed_lambda"] = fixed_lambda
    result["lambda_grid"] = lambda_grid
    result["train_pairs"] = int(len(X_train))
    result["source_fineweb_records"] = args.fineweb_records
    result["source_scifact_records"] = args.scifact_records
    result["evaluation_note"] = "Predicted lambdas are snapped to the nearest saved SciFact sweep lambda."

    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    export_kshot_latex(args.output_path, os.path.splitext(args.output_path)[0] + "_kshot_table.tex")
    save_run_metadata(
        f"{args.output_path}.metadata.json",
        args,
        extra={"lambda_grid": lambda_grid, "fixed_lambda": fixed_lambda},
    )
    LOGGER.info("saved: %s", args.output_path)


if __name__ == "__main__":
    main()

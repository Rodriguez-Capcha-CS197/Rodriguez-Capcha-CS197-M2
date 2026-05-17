"""Paper-grade bottleneck ablation: FineWeb -> SciFact with live lambda_fn retrieval."""

from __future__ import annotations

import argparse
import copy
import json
import logging
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from beir import util
from beir.datasets.data_loader import GenericDataLoader

from configs.metadata import save_run_metadata
from scripts.reporting import export_kshot_latex
from shared.beir_data import resolve_beir_dataset_path
from shared.beir_scoring import scifact_ndcg
from shared.constants import LAMBDA_GRID
from shared.embedding import MiniLMEmbedder
from shared.logging_utils import configure_logging
from shared.predictor import LambdaPredictor
from shared.schemas import QueryExample
from shared.scoring import ensure_1d
from shared.segments import SegmentBuildConfig, build_segments_from_docs
from ska_agent.core.pricing import PricingEngine
from ska_agent.core.structures import Segment


LOGGER = logging.getLogger(__name__)


# Download SciFact when requested.
def _download_scifact(data_dir: str) -> str:
    url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip"
    return util.download_and_unzip(url, data_dir)


# Convert a FineWeb JSONL query file into examples.
def _load_fineweb_queries(path: str, max_queries: int | None) -> tuple[list[str], list[str], list[QueryExample]]:
    passages: dict[str, str] = {}
    examples: list[QueryExample] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            passage_id = str(row["passage_id"])
            passage_text = str(row["passage_text"])
            passages[passage_id] = passage_text
            for i, q in enumerate(row.get("queries", [])):
                examples.append(QueryExample(qid=f"{passage_id}_q{i}", query=str(q), relevant_doc_ids=[passage_id]))
                if max_queries is not None and len(examples) >= max_queries:
                    doc_ids = list(passages.keys())
                    doc_texts = [passages[d] for d in doc_ids]
                    return doc_ids, doc_texts, examples
    doc_ids = list(passages.keys())
    doc_texts = [passages[d] for d in doc_ids]
    return doc_ids, doc_texts, examples


# Build SciFact query examples from BEIR data.
def _prepare_scifact_examples(queries: dict[str, str], qrels: dict[str, dict[str, int]]) -> list[QueryExample]:
    rows: list[QueryExample] = []
    for qid, query_text in queries.items():
        rel = qrels.get(qid, {})
        if not rel:
            continue
        rows.append(QueryExample(qid=qid, query=str(query_text), relevant_doc_ids=list(rel.keys())))
    return rows


# Compute retrieval F1 against gold relevant doc ids.
def _retrieval_f1(retrieved_doc_ids: list[str], relevant_doc_ids: list[str]) -> float:
    retrieved = set(retrieved_doc_ids)
    relevant = set(relevant_doc_ids)
    tp = len(retrieved & relevant)
    precision = 0.0 if not retrieved else tp / len(retrieved)
    recall = 0.0 if not relevant else tp / len(relevant)
    if precision + recall == 0.0:
        return 0.0
    return float(2.0 * precision * recall / (precision + recall))


# Sweep lambda values for one query and return best lambda.
def _best_lambda_for_query(
    query_text: str,
    relevant_doc_ids: list[str],
    engines: dict[float, PricingEngine],
    segment_to_doc_id: dict[int, str],
) -> float:
    best_lambda = None
    best_score = -float("inf")
    best_k = None
    for lam in sorted(engines.keys()):
        result = engines[lam].retrieve(query_text, verbose=False)
        retrieved_doc_ids = [
            segment_to_doc_id[int(seg.start_idx)]
            for seg in result.segments
            if int(seg.start_idx) in segment_to_doc_id
        ]
        score = _retrieval_f1(retrieved_doc_ids, relevant_doc_ids)
        k = len(result.segments)
        if score > best_score:
            best_score = score
            best_lambda = lam
            best_k = k
        elif abs(score - best_score) <= 1e-9:
            if best_k is None or k < best_k or (k == best_k and lam < best_lambda):
                best_lambda = lam
                best_k = k
    return float(best_lambda if best_lambda is not None else 1.0)


# Train a predictor in log-lambda space and keep best validation checkpoint.
def _train_predictor(
    X: np.ndarray,
    y: np.ndarray,
    seed: int,
    hidden_dim: int,
    epochs: int,
    lr: float,
) -> LambdaPredictor:
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


# Predict a positive lambda from a query string.
def _predict_lambda(model: LambdaPredictor, embedder: MiniLMEmbedder, query_text: str) -> float:
    q_emb = ensure_1d(embedder.embed_single(query_text)).astype(np.float32)
    with torch.no_grad():
        log_lam = model(torch.tensor(q_emb, dtype=torch.float32).unsqueeze(0)).item()
    return float(np.exp(log_lam))


# Fine-tune a trained lambda predictor on small in-domain k-shot labels.
def _few_shot_finetune(
    base_model: LambdaPredictor,
    X_shot: np.ndarray,
    y_shot: np.ndarray,
    epochs: int = 80,
    lr: float = 5e-4,
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


# Evaluate one query by running live PricingEngine retrieval with selected lambda.
def _run_query_with_lambda(
    qid: str,
    query_text: str,
    lam: float,
    segments: list[Segment],
    embedder: MiniLMEmbedder,
    segment_to_doc_id: dict[int, str],
    qrels: dict[str, dict[str, int]],
) -> float:
    engine = PricingEngine(
        segments=segments,
        embed_fn=embedder.embed_single,
        lambda_sparsity=float(lam),
        eta_redundancy=0.0,
        max_segments=5,
    )
    result = engine.retrieve(query_text, verbose=False)
    retrieved_doc_ids = [
        segment_to_doc_id[int(seg.start_idx)]
        for seg in result.segments
        if int(seg.start_idx) in segment_to_doc_id
    ]
    return scifact_ndcg(
        retrieved_segments=result.segments,
        query_id=qid,
        qrels=qrels,
        segment_to_doc_id=segment_to_doc_id,
        k=10,
    )


# Run SciFact ablation conditions and return aggregate metrics.
def _run_ablation(
    scifact_examples: list[QueryExample],
    scifact_segments: list[Segment],
    scifact_seg_to_doc: dict[int, str],
    embedder: MiniLMEmbedder,
    train_model: LambdaPredictor,
    fixed_lambda: float,
    lambda_grid: list[float],
    k_shot: int,
    seed: int,
    qrels: dict[str, dict[str, int]],
) -> dict:
    engines = {
        lam: PricingEngine(
            segments=scifact_segments,
            embed_fn=embedder.embed_single,
            lambda_sparsity=float(lam),
            eta_redundancy=0.0,
            max_segments=5,
        )
        for lam in lambda_grid
    }

    oracle_lambda_by_qid: dict[str, float] = {}
    for ex in scifact_examples:
        oracle_lambda_by_qid[ex.qid] = _best_lambda_for_query(ex.query, ex.relevant_doc_ids, engines, scifact_seg_to_doc)

    rng = np.random.default_rng(seed)
    qids = [ex.qid for ex in scifact_examples]
    rng.shuffle(qids)
    shot_qids = set(qids[: min(k_shot, len(qids))])

    shot_X: list[np.ndarray] = []
    shot_y: list[float] = []
    for ex in scifact_examples:
        if ex.qid in shot_qids:
            shot_X.append(ensure_1d(embedder.embed_single(ex.query)).astype(np.float32))
            shot_y.append(oracle_lambda_by_qid[ex.qid])

    ft_model = _few_shot_finetune(
        train_model,
        np.asarray(shot_X, dtype=np.float32) if shot_X else np.empty((0, train_model.net[0].in_features), dtype=np.float32),
        np.asarray(shot_y, dtype=np.float32) if shot_y else np.empty((0,), dtype=np.float32),
    )

    oracle_by_qid: dict[str, float] = {}
    fixed_by_qid: dict[str, float] = {}
    zero_by_qid: dict[str, float] = {}
    few_by_qid: dict[str, float] = {}

    for ex in scifact_examples:
        oracle_by_qid[ex.qid] = _run_query_with_lambda(
            ex.qid, ex.query, oracle_lambda_by_qid[ex.qid],
            scifact_segments, embedder, scifact_seg_to_doc, qrels,
        )
        fixed_by_qid[ex.qid] = _run_query_with_lambda(
            ex.qid, ex.query, fixed_lambda,
            scifact_segments, embedder, scifact_seg_to_doc, qrels,
        )
        zero_lam = _predict_lambda(train_model, embedder, ex.query)
        zero_by_qid[ex.qid] = _run_query_with_lambda(
            ex.qid, ex.query, zero_lam,
            scifact_segments, embedder, scifact_seg_to_doc, qrels,
        )
        if ex.qid in shot_qids:
            continue
        ft_lam = _predict_lambda(ft_model, embedder, ex.query)
        few_by_qid[ex.qid] = _run_query_with_lambda(
            ex.qid, ex.query, ft_lam,
            scifact_segments, embedder, scifact_seg_to_doc, qrels,
        )

    held_out_qids = [ex.qid for ex in scifact_examples if ex.qid not in shot_qids]

    # All-query means (for reporting).
    oracle_all = float(np.mean(list(oracle_by_qid.values()))) if oracle_by_qid else 0.0
    fixed_all = float(np.mean(list(fixed_by_qid.values()))) if fixed_by_qid else 0.0
    zero_all = float(np.mean(list(zero_by_qid.values()))) if zero_by_qid else 0.0

    # Held-out-only means for apples-to-apples recovery comparison.
    oracle_held = float(np.mean([oracle_by_qid[q] for q in held_out_qids])) if held_out_qids else 0.0
    zero_held = float(np.mean([zero_by_qid[q] for q in held_out_qids])) if held_out_qids else 0.0
    few_held = float(np.mean([few_by_qid[q] for q in held_out_qids])) if held_out_qids else 0.0

    gap = oracle_held - zero_held
    recovery_pct = float(100.0 * (few_held - zero_held) / gap) if gap > 1e-9 else 0.0

    print(f"\n=== Fine-tuning Recovery (FineWeb → SciFact, k={k_shot}) ===")
    print(f"Oracle nDCG@10 (all):           {oracle_all:.3f}")
    print(f"Fixed-lambda nDCG@10 (all):     {fixed_all:.3f}")
    print(f"MLP zero-shot nDCG@10 (all):    {zero_all:.3f}")
    print(f"--- held-out queries only (n={len(held_out_qids)}) ---")
    print(f"Oracle nDCG@10:                 {oracle_held:.3f}")
    print(f"MLP zero-shot nDCG@10:          {zero_held:.3f}")
    print(f"MLP {k_shot}-shot fine-tuned nDCG@10:   {few_held:.3f}")
    print(f"Oracle gap:                     {gap:.3f}")
    print(f"Recovery:                       {recovery_pct:.1f}% of oracle gap")

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
        "num_scifact_queries": int(len(scifact_examples)),
    }


# Build FineWeb training labels with the shared segment builder + lambda sweeps.
def _build_fineweb_labels(
    fineweb_queries_path: str,
    max_fineweb_queries: int | None,
    embedder: MiniLMEmbedder,
    lambda_grid: list[float],
    segment_config: SegmentBuildConfig,
) -> tuple[np.ndarray, np.ndarray]:
    doc_ids, doc_texts, examples = _load_fineweb_queries(fineweb_queries_path, max_fineweb_queries)
    segments, seg_to_doc = build_segments_from_docs(
        doc_ids,
        doc_texts,
        embedder,
        config=segment_config,
    )
    engines = {
        lam: PricingEngine(
            segments=segments,
            embed_fn=embedder.embed_single,
            lambda_sparsity=float(lam),
            eta_redundancy=0.0,
            max_segments=5,
        )
        for lam in lambda_grid
    }
    X = []
    y = []
    for i, ex in enumerate(examples, start=1):
        best_lam = _best_lambda_for_query(ex.query, ex.relevant_doc_ids, engines, seg_to_doc)
        X.append(ensure_1d(embedder.embed_single(ex.query)).astype(np.float32))
        y.append(best_lam)
        if i % 200 == 0:
            print(f"fineweb labels: {i}/{len(examples)}")
    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.float32)


# Main entrypoint for the full paper ablation.
def main() -> None:
    configure_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument("--fineweb-queries-path", default="outputs/fineweb_queries.jsonl")
    parser.add_argument("--max-fineweb-queries", type=int, default=25000)
    parser.add_argument("--scifact-data-dir", default="outputs/beir_data")
    parser.add_argument("--download-scifact", action="store_true")
    parser.add_argument("--scifact-split", default="test")
    parser.add_argument("--embed-model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--lambda-grid", default=",".join(str(x) for x in LAMBDA_GRID))
    parser.add_argument("--k-shot", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--min-sentence-len", type=int, default=20)
    parser.add_argument("--min-segment-size", type=int, default=2)
    parser.add_argument("--max-segment-size", type=int, default=15)
    parser.add_argument("--lookback-k", type=int, default=50)
    parser.add_argument("--segment-strategy", default="geometry_sentence")
    parser.add_argument("--output-path", default="outputs/paper_bottleneck_ablation_scifact.json")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    lambda_grid = sorted({float(x.strip()) for x in args.lambda_grid.split(",") if x.strip()})
    embedder = MiniLMEmbedder(args.embed_model)
    segment_config = SegmentBuildConfig(
        strategy=args.segment_strategy,
        min_sentence_len=args.min_sentence_len,
        min_segment_size=args.min_segment_size,
        max_segment_size=args.max_segment_size,
        lookback_k=args.lookback_k,
    )

    LOGGER.info("building FineWeb labels")
    X_train, y_train = _build_fineweb_labels(
        args.fineweb_queries_path,
        args.max_fineweb_queries,
        embedder,
        lambda_grid,
        segment_config=segment_config,
    )
    LOGGER.info("fineweb training pairs: %d", len(X_train))

    model = _train_predictor(X_train, y_train, seed=args.seed, hidden_dim=64, epochs=200, lr=1e-3)
    fixed_lambda = float(np.median(y_train))

    LOGGER.info("loading SciFact")
    scifact_path = resolve_beir_dataset_path(
        args.scifact_data_dir,
        "scifact",
        args.scifact_split,
        allow_download=args.download_scifact,
        downloader=_download_scifact,
    )
    corpus, queries, qrels = GenericDataLoader(scifact_path).load(split=args.scifact_split)
    scifact_doc_ids = []
    scifact_doc_texts = []
    for doc_id, doc in corpus.items():
        title = str(doc.get("title", "")).strip()
        text = str(doc.get("text", "")).strip()
        merged = f"{title} {text}".strip()
        if not merged:
            continue
        scifact_doc_ids.append(str(doc_id))
        scifact_doc_texts.append(merged)

    LOGGER.info("building SciFact geometry segments")
    scifact_segments, scifact_seg_to_doc = build_segments_from_docs(
        scifact_doc_ids,
        scifact_doc_texts,
        embedder,
        config=segment_config,
    )
    scifact_examples = _prepare_scifact_examples(queries, qrels)

    LOGGER.info("running ablation")
    result = _run_ablation(
        scifact_examples=scifact_examples,
        scifact_segments=scifact_segments,
        scifact_seg_to_doc=scifact_seg_to_doc,
        embedder=embedder,
        train_model=model,
        fixed_lambda=fixed_lambda,
        lambda_grid=lambda_grid,
        k_shot=args.k_shot,
        seed=args.seed,
        qrels=qrels,
    )
    result["fixed_lambda"] = fixed_lambda
    result["lambda_grid"] = lambda_grid
    result["train_pairs"] = int(len(X_train))
    result["segment_strategy"] = segment_config.strategy
    result["segment_config"] = segment_config.to_dict()
    result["num_scifact_segments"] = int(len(scifact_segments))

    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    export_kshot_latex(args.output_path, os.path.splitext(args.output_path)[0] + "_kshot_table.tex")
    save_run_metadata(
        f"{args.output_path}.metadata.json",
        args,
        extra={"segment_config": segment_config.to_dict(), "lambda_grid": lambda_grid},
    )
    LOGGER.info("saved: %s", args.output_path)


if __name__ == "__main__":
    main()

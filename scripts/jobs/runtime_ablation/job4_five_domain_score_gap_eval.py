"""Five-domain runtime score-gap evaluation.

This module evaluates the runtime score-gap heuristic against a fixed-lambda
baseline on BEIR sweep records. It is intentionally isolated from the existing
MLP evaluation pipeline.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from beir.datasets.data_loader import GenericDataLoader

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from configs.metadata import save_run_metadata
from shared.beir_data import resolve_beir_dataset_path
from shared.beir_scoring import beir_ndcg, build_beir_segments
from shared.constants import EPS, MAX_SEGMENTS
from shared.embedding import MiniLMEmbedder
from shared.logging_utils import configure_logging
from shared.segments import SegmentBuildConfig


LOGGER = logging.getLogger(__name__)

DEFAULT_DOMAINS = ["scifact", "fiqa", "nfcorpus", "arguana", "trec-covid"]
METHODS = ["fixed", "runtime_score_gap"]
PRIMARY_METRICS = ["ndcg_at_10", "precision_returned", "num_returned_docs"]


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


def _read_json_rows(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    return data if isinstance(data, list) else [data]


def _write_jsonl(path: str, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def _group_by_query(records: list[dict]) -> dict[str, list[dict]]:
    rows_by_qid: dict[str, list[dict]] = defaultdict(list)
    for row in records:
        rows_by_qid[str(row["question_id"])].append(row)
    return dict(rows_by_qid)


def _dedupe_doc_ids(doc_ids) -> list[str]:
    return list(dict.fromkeys(str(doc_id) for doc_id in doc_ids))


def _metrics_from_doc_ids(ndcg_at_10: float, retrieved_doc_ids, relevant_doc_ids) -> dict:
    retrieved_doc_ids = _dedupe_doc_ids(retrieved_doc_ids)
    relevant_doc_ids = {str(doc_id) for doc_id in relevant_doc_ids}
    hit_count = sum(1 for doc_id in retrieved_doc_ids if doc_id in relevant_doc_ids)
    precision = hit_count / len(retrieved_doc_ids) if retrieved_doc_ids else 0.0
    return {
        "ndcg_at_10": float(ndcg_at_10),
        "precision_returned": float(precision),
        "num_returned_docs": float(len(retrieved_doc_ids)),
        "hit_count": int(hit_count),
        "retrieved_doc_ids": retrieved_doc_ids,
        "relevant_doc_ids": sorted(relevant_doc_ids),
    }


def _metrics_from_sweep_row(row: dict) -> dict:
    return _metrics_from_doc_ids(
        ndcg_at_10=float(row["ndcg_at_10"]),
        retrieved_doc_ids=row.get("retrieved_doc_ids", []),
        relevant_doc_ids=row.get("relevant_doc_ids", []),
    )


def _nearest_lambda_row(rows: list[dict], fixed_lambda: float) -> dict:
    return min(rows, key=lambda row: abs(float(row["lambda"]) - float(fixed_lambda)))


def _fixed_lambda_from_training_records(fineweb_records: str, marco_records: str) -> float:
    lambdas = []
    for path in [fineweb_records, marco_records]:
        records = _read_json_rows(path)
        lambdas.extend(float(row["lambda"]) for row in records if row.get("is_optimal"))
    if not lambdas:
        raise ValueError(
            "Could not infer fixed lambda because no optimal rows were found in "
            f"{fineweb_records!r} or {marco_records!r}. Pass --fixed-lambda explicitly."
        )
    return float(np.median(np.asarray(lambdas, dtype=np.float32)))


def _assert_segment_config(records: list[dict], expected_config: SegmentBuildConfig, records_path: str) -> None:
    strategies = {row.get("segment_strategy") for row in records if row.get("segment_strategy")}
    if strategies and strategies != {expected_config.strategy}:
        raise ValueError(
            f"{records_path} was generated with segment strategies {sorted(strategies)}, "
            f"but this run requested {expected_config.strategy!r}."
        )

    configs = [row.get("segment_config") for row in records if row.get("segment_config")]
    if not configs:
        LOGGER.warning("%s has no segment_config metadata; assuming current config", records_path)
        return

    expected = expected_config.to_dict()
    mismatches = []
    for key, expected_value in expected.items():
        seen = {cfg.get(key) for cfg in configs}
        if seen != {expected_value}:
            mismatches.append(f"{key}: saw {sorted(seen)!r}, expected {expected_value!r}")
    if mismatches:
        raise ValueError(f"{records_path} segment_config mismatch: " + "; ".join(mismatches))


def _runtime_score_gap_rows(
    records_by_query: dict[str, list[dict]],
    segments,
    segment_to_doc_id: dict[int, str],
    qrels: dict[str, dict[str, int]],
    embedder: MiniLMEmbedder,
    candidate_k: int,
    max_segments: int,
) -> list[dict]:
    corpus_embs = np.asarray([segment.vector for segment in segments], dtype=np.float32)
    corpus_norms = np.linalg.norm(corpus_embs, axis=1)

    qids = list(records_by_query.keys())
    query_texts = [records_by_query[qid][0]["query"] for qid in qids]
    query_embs = np.asarray(embedder.embed(query_texts), dtype=np.float32)

    rows = []
    for qid, query, query_emb in zip(qids, query_texts, query_embs):
        query_emb = np.asarray(query_emb, dtype=np.float32).reshape(-1)
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
        ndcg = beir_ndcg(
            retrieved_segments=selected_segments,
            query_id=qid,
            qrels=qrels,
            segment_to_doc_id=segment_to_doc_id,
            k=10,
        )
        metrics = _metrics_from_doc_ids(
            ndcg_at_10=ndcg,
            retrieved_doc_ids=retrieved_doc_ids,
            relevant_doc_ids=records_by_query[qid][0].get("relevant_doc_ids", []),
        )
        rows.append(
            {
                "question_id": qid,
                "query": query,
                "method": "runtime_score_gap",
                "chosen_lambda": None,
                "runtime_score_gap_k": int(len(selected_segments)),
                "runtime_candidate_k": int(min(candidate_k, len(scores))),
                **metrics,
            }
        )
    return rows


def _fixed_rows(records_by_query: dict[str, list[dict]], fixed_lambda: float) -> list[dict]:
    rows = []
    for qid, sweep_rows in records_by_query.items():
        selected = _nearest_lambda_row(sweep_rows, fixed_lambda)
        metrics = _metrics_from_sweep_row(selected)
        rows.append(
            {
                "question_id": qid,
                "query": selected["query"],
                "method": "fixed",
                "chosen_lambda": float(selected["lambda"]),
                **metrics,
            }
        )
    return rows


def _aggregate_query_rows(query_rows: list[dict]) -> dict:
    return {
        metric: float(np.mean([float(row[metric]) for row in query_rows])) if query_rows else 0.0
        for metric in PRIMARY_METRICS
    }


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


def _bootstrap_summary(domain: str, fixed_rows: list[dict], runtime_rows: list[dict], n_samples: int, seed: int) -> dict:
    fixed_by_qid = {row["question_id"]: row for row in fixed_rows}
    runtime_by_qid = {row["question_id"]: row for row in runtime_rows}
    common_qids = [qid for qid in runtime_by_qid if qid in fixed_by_qid]

    summary = {
        "domain": domain,
        "n_queries": len(common_qids),
        "fixed": {},
        "runtime_score_gap": {},
        "runtime_score_gap_diff_vs_fixed": {},
    }
    for offset, metric in enumerate(PRIMARY_METRICS):
        fixed_values = [float(fixed_by_qid[qid][metric]) for qid in common_qids]
        runtime_values = [float(runtime_by_qid[qid][metric]) for qid in common_qids]
        summary["fixed"][metric] = _bootstrap_mean_ci(fixed_values, n_samples, seed + offset)
        summary["runtime_score_gap"][metric] = _bootstrap_mean_ci(runtime_values, n_samples, seed + 20 + offset)
        summary["runtime_score_gap_diff_vs_fixed"][metric] = _paired_bootstrap_diff_ci(
            runtime_values,
            fixed_values,
            n_samples,
            seed + 40 + offset,
        )
    return summary


def _format_ci(item: dict) -> str:
    return f"{item['mean']:.3f} [{item['ci_low']:.3f}, {item['ci_high']:.3f}]"


def _format_delta(item: dict) -> str:
    return (
        f"{item['mean_diff']:+.3f} "
        f"[{item['ci_low']:+.3f}, {item['ci_high']:+.3f}], "
        f"p={item['p_two_sided']:.3f}"
    )


def _paper_table_rows(bootstrap_summaries: list[dict]) -> list[dict]:
    rows = []
    for summary in bootstrap_summaries:
        domain = summary["domain"]
        for method in METHODS:
            row = {
                "domain": domain,
                "method": method,
                "n_queries": summary["n_queries"],
                "nDCG@10 95% CI": _format_ci(summary[method]["ndcg_at_10"]),
                "precision@returned 95% CI": _format_ci(summary[method]["precision_returned"]),
                "Avg returned 95% CI": _format_ci(summary[method]["num_returned_docs"]),
            }
            if method == "fixed":
                row["Delta nDCG vs fixed"] = "--"
                row["Delta precision vs fixed"] = "--"
                row["Delta returned vs fixed"] = "--"
            else:
                deltas = summary[f"{method}_diff_vs_fixed"]
                row["Delta nDCG vs fixed"] = _format_delta(deltas["ndcg_at_10"])
                row["Delta precision vs fixed"] = _format_delta(deltas["precision_returned"])
                row["Delta returned vs fixed"] = _format_delta(deltas["num_returned_docs"])
            rows.append(row)
    return rows


def _write_csv(path: str, rows: list[dict]) -> None:
    if not rows:
        return
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _latex_escape(value) -> str:
    text = str(value)
    replacements = {
        "\\": "\\textbackslash{}",
        "&": "\\&",
        "%": "\\%",
        "$": "\\$",
        "#": "\\#",
        "_": "\\_",
        "{": "\\{",
        "}": "\\}",
        "~": "\\textasciitilde{}",
        "^": "\\textasciicircum{}",
    }
    for raw, escaped in replacements.items():
        text = text.replace(raw, escaped)
    return text


def _write_latex_table(path: str, rows: list[dict]) -> None:
    if not rows:
        return
    columns = list(rows[0].keys())
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\\begin{tabular}{" + "l" * len(columns) + "}\n")
        handle.write("\\toprule\n")
        handle.write(" & ".join(_latex_escape(col) for col in columns) + " \\\\\n")
        handle.write("\\midrule\n")
        for row in rows:
            handle.write(" & ".join(_latex_escape(row[col]) for col in columns) + " \\\\\n")
        handle.write("\\bottomrule\n")
        handle.write("\\end{tabular}\n")


def _domain_records_path(records_dir: str, domain: str) -> str:
    return os.path.join(records_dir, f"{domain}_sweep_records.json")


def _parse_domains(args) -> list[dict]:
    domains = []
    for raw in [item.strip() for item in args.domains.split(",") if item.strip()]:
        parts = raw.split(":")
        name = parts[0]
        records_path = parts[1] if len(parts) > 1 and parts[1] else _domain_records_path(args.records_dir, name)
        data_path = parts[2] if len(parts) > 2 and parts[2] else os.path.join(args.beir_data_dir, name)
        split = parts[3] if len(parts) > 3 and parts[3] else args.split
        domains.append({"name": name, "records_path": records_path, "data_path": data_path, "split": split})
    return domains


def _evaluate_domain(
    domain: dict,
    fixed_lambda: float,
    embedder: MiniLMEmbedder,
    segment_config: SegmentBuildConfig,
    candidate_k: int,
    max_segments: int,
    max_queries: int,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> tuple[dict, list[dict], dict]:
    records = _read_json_rows(domain["records_path"])
    _assert_segment_config(records, segment_config, domain["records_path"])
    records_by_query = _group_by_query(records)
    if max_queries and max_queries > 0:
        records_by_query = dict(list(records_by_query.items())[:max_queries])

    fixed_rows = _fixed_rows(records_by_query, fixed_lambda)

    data_path = resolve_beir_dataset_path(domain["data_path"], domain["name"], domain["split"])
    corpus, _, qrels = GenericDataLoader(data_path).load(split=domain["split"])
    segments, segment_to_doc_id = build_beir_segments(corpus, embedder, segment_config=segment_config)
    runtime_rows = _runtime_score_gap_rows(
        records_by_query=records_by_query,
        segments=segments,
        segment_to_doc_id=segment_to_doc_id,
        qrels=qrels,
        embedder=embedder,
        candidate_k=candidate_k,
        max_segments=max_segments,
    )

    run_row = {
        "domain": domain["name"],
        "fixed_lambda": float(fixed_lambda),
        "n_queries": int(len(records_by_query)),
        "num_corpus_segments": int(len(segments)),
    }
    for method, rows in [("fixed", fixed_rows), ("runtime_score_gap", runtime_rows)]:
        aggregates = _aggregate_query_rows(rows)
        for metric, value in aggregates.items():
            run_row[f"{method}_{metric}"] = value
        run_row[f"{method}_avg_returned"] = aggregates["num_returned_docs"]

    query_rows = []
    for method_rows in [fixed_rows, runtime_rows]:
        for row in method_rows:
            query_rows.append({"domain": domain["name"], **row})

    bootstrap = _bootstrap_summary(
        domain=domain["name"],
        fixed_rows=fixed_rows,
        runtime_rows=runtime_rows,
        n_samples=bootstrap_samples,
        seed=bootstrap_seed,
    )
    return run_row, query_rows, bootstrap


def main() -> None:
    configure_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument("--domains", default=",".join(DEFAULT_DOMAINS))
    parser.add_argument("--records-dir", default="outputs")
    parser.add_argument("--beir-data-dir", default="outputs/beir_data")
    parser.add_argument("--fineweb-records", default="outputs/fineweb_labeled.json")
    parser.add_argument("--marco-records", default="outputs/marco_labeled.json")
    parser.add_argument("--fixed-lambda", type=float, default=None)
    parser.add_argument("--split", default="test")
    parser.add_argument("--embed-model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--output-dir", default="outputs/runtime_ablation/five_domain_score_gap")
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    parser.add_argument("--runtime-candidate-k", type=int, default=50)
    parser.add_argument("--runtime-max-segments", type=int, default=MAX_SEGMENTS)
    parser.add_argument("--max-queries", type=int, default=0)
    parser.add_argument("--segment-strategy", default="geometry_sentence")
    parser.add_argument("--min-sentence-len", type=int, default=20)
    parser.add_argument("--min-segment-size", type=int, default=2)
    parser.add_argument("--max-segment-size", type=int, default=15)
    parser.add_argument("--lookback-k", type=int, default=50)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    domains = _parse_domains(args)
    segment_config = SegmentBuildConfig(
        strategy=args.segment_strategy,
        min_sentence_len=args.min_sentence_len,
        min_segment_size=args.min_segment_size,
        max_segment_size=args.max_segment_size,
        lookback_k=args.lookback_k,
    )
    fixed_lambda = (
        float(args.fixed_lambda)
        if args.fixed_lambda is not None
        else _fixed_lambda_from_training_records(args.fineweb_records, args.marco_records)
    )
    embedder = MiniLMEmbedder(args.embed_model)

    run_rows = []
    query_rows = []
    bootstrap_summaries = []
    for domain in domains:
        LOGGER.info("evaluating runtime score-gap on %s", domain["name"])
        run_row, domain_query_rows, bootstrap = _evaluate_domain(
            domain=domain,
            fixed_lambda=fixed_lambda,
            embedder=embedder,
            segment_config=segment_config,
            candidate_k=args.runtime_candidate_k,
            max_segments=args.runtime_max_segments,
            max_queries=args.max_queries,
            bootstrap_samples=args.bootstrap_samples,
            bootstrap_seed=args.bootstrap_seed,
        )
        run_rows.append(run_row)
        query_rows.extend(domain_query_rows)
        bootstrap_summaries.append(bootstrap)

    with open(os.path.join(args.output_dir, "five_domain_score_gap_runs.json"), "w", encoding="utf-8") as handle:
        json.dump(run_rows, handle, indent=2)
    _write_jsonl(os.path.join(args.output_dir, "five_domain_score_gap_query_metrics.jsonl"), query_rows)
    with open(
        os.path.join(args.output_dir, "five_domain_score_gap_bootstrap_summary.json"),
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(bootstrap_summaries, handle, indent=2)

    table_rows = _paper_table_rows(bootstrap_summaries)
    _write_csv(os.path.join(args.output_dir, "five_domain_score_gap_paper_table.csv"), table_rows)
    _write_latex_table(os.path.join(args.output_dir, "five_domain_score_gap_paper_table.tex"), table_rows)

    save_run_metadata(
        os.path.join(args.output_dir, "five_domain_score_gap_metadata.json"),
        args,
        extra={
            "domains": domains,
            "fixed_lambda": fixed_lambda,
            "methods": METHODS,
            "primary_metrics": PRIMARY_METRICS,
            "segment_config": segment_config.to_dict(),
        },
    )
    LOGGER.info("saved five-domain runtime score-gap outputs to %s", args.output_dir)


if __name__ == "__main__":
    main()

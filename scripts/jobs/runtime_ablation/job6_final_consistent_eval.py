"""Final consistency pass for runtime ablation paper tables.

This job does not generate sweeps or train models. It consumes saved BEIR
lambda-sweep records and optional existing policy-query metric files, then
recomputes the corrected final tables with consistent oracle tie-breaking and
shared BEIR nDCG scoring for score-gap retrieval.
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
from scripts.jobs.runtime_ablation.job4_five_domain_score_gap_eval import (
    score_gap_cutoff_indices as job4_score_gap_cutoff_indices,
)
from scripts.jobs.runtime_ablation.job5_learned_runtime_feature_eval import (
    score_gap_cutoff_indices as job5_score_gap_cutoff_indices,
)


LOGGER = logging.getLogger(__name__)

DEFAULT_DOMAINS = ["scifact", "fiqa", "nfcorpus", "arguana", "trec-covid"]
METHOD_ORDER = [
    "fixed",
    "plain",
    "hybrid",
    "covariance",
    "runtime_score_gap",
    "ndcg_oracle",
    "precision_oracle",
]
TABLE_METRICS = [
    "ndcg_at_10",
    "precision_returned",
    "num_returned_segments",
    "num_returned_docs",
]


def _read_json_rows(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    return data if isinstance(data, list) else [data]


def _read_jsonl(path: str) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _write_jsonl(path: str, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


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


def _group_by_query(records: list[dict]) -> dict[str, list[dict]]:
    rows_by_qid: dict[str, list[dict]] = defaultdict(list)
    for row in records:
        rows_by_qid[str(row["question_id"])].append(row)
    return dict(rows_by_qid)


def _dedupe_doc_ids(doc_ids) -> list[str]:
    return list(dict.fromkeys(str(doc_id) for doc_id in doc_ids))


def _metrics_from_retrieval(
    ndcg_at_10: float,
    retrieved_doc_ids,
    relevant_doc_ids,
    num_segments: int,
) -> dict:
    retrieved_doc_ids = [str(doc_id) for doc_id in retrieved_doc_ids]
    unique_doc_ids = _dedupe_doc_ids(retrieved_doc_ids)
    relevant = {str(doc_id) for doc_id in relevant_doc_ids}
    doc_hits = sum(1 for doc_id in unique_doc_ids if doc_id in relevant)
    segment_hits = sum(1 for doc_id in retrieved_doc_ids if doc_id in relevant)
    precision = doc_hits / len(unique_doc_ids) if unique_doc_ids else 0.0
    segment_precision = segment_hits / int(num_segments) if int(num_segments) > 0 else 0.0
    return {
        "ndcg_at_10": float(ndcg_at_10),
        "precision_returned": float(precision),
        "precision_returned_segments": float(segment_precision),
        "num_returned_segments": float(num_segments),
        "num_returned_docs": float(len(unique_doc_ids)),
        "hit_count_docs": int(doc_hits),
        "hit_count_segments": int(segment_hits),
        "retrieved_doc_ids": unique_doc_ids,
        "retrieved_segment_doc_ids": retrieved_doc_ids,
        "relevant_doc_ids": sorted(relevant),
    }


def _metrics_from_sweep_row(row: dict) -> dict:
    retrieved_doc_ids = row.get("retrieved_doc_ids", [])
    return _metrics_from_retrieval(
        ndcg_at_10=float(row["ndcg_at_10"]),
        retrieved_doc_ids=retrieved_doc_ids,
        relevant_doc_ids=row.get("relevant_doc_ids", []),
        num_segments=int(row.get("num_segments", len(retrieved_doc_ids))),
    )


def _nearest_lambda_row(rows: list[dict], lam: float) -> dict:
    return min(rows, key=lambda row: abs(float(row["lambda"]) - float(lam)))


def _old_first_tie_ndcg_oracle(rows: list[dict]) -> dict:
    """Reproduce the old first-tie behavior for sanity diagnostics only."""
    best = rows[0]
    best_ndcg = float(best["ndcg_at_10"])
    for row in rows[1:]:
        ndcg = float(row["ndcg_at_10"])
        if ndcg > best_ndcg:
            best = row
            best_ndcg = ndcg
    return best


def _sparse_ndcg_oracle(rows: list[dict]) -> dict:
    # nDCG ties are broken toward fewer admitted segments to avoid treating
    # equally ranked but more permissive retrieval as oracle-superior.
    return max(rows, key=lambda row: (float(row["ndcg_at_10"]), -int(row.get("num_segments", 0))))


def _precision_oracle(rows: list[dict]) -> dict:
    return max(
        rows,
        key=lambda row: (
            float(_metrics_from_sweep_row(row)["precision_returned"]),
            float(row["ndcg_at_10"]),
            -int(row.get("num_segments", 0)),
        ),
    )


def _policy_row_from_sweep(
    domain: str,
    qid: str,
    method: str,
    selected_row: dict,
    chosen_lambda: float | None,
) -> dict:
    return {
        "domain": domain,
        "question_id": qid,
        "query": selected_row["query"],
        "method": method,
        "chosen_lambda": None if chosen_lambda is None else float(chosen_lambda),
        **_metrics_from_sweep_row(selected_row),
    }


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


def _load_policy_metric_rows(paths: list[str]) -> dict[tuple[str, str, str], list[dict]]:
    rows_by_key: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for path in paths:
        if not path or not os.path.exists(path):
            continue
        LOGGER.info("loading optional policy query metrics: %s", path)
        for row in _read_jsonl(path):
            method = str(row.get("method", ""))
            if method not in {"plain", "hybrid", "covariance"}:
                continue
            if row.get("chosen_lambda") is None:
                continue
            domain = str(row.get("domain", "")).strip().lower().replace(" ", "-")
            qid = str(row.get("question_id", ""))
            if domain and qid:
                rows_by_key[(domain, method, qid)].append(row)
    return dict(rows_by_key)


def _auto_policy_metric_paths(output_root: str) -> list[str]:
    candidates = [
        os.path.join(output_root, "week9", "policy_query_metrics.jsonl"),
        os.path.join(output_root, "runtime_score_gap_ablation", "runtime_score_gap_policy_query_metrics.jsonl"),
        os.path.join(
            output_root,
            "runtime_ablation",
            "learned_runtime_feature",
            "learned_runtime_feature_query_metrics.jsonl",
        ),
    ]
    return [path for path in candidates if os.path.exists(path)]


def _policy_rows_from_existing_metrics(
    domain: str,
    records_by_query: dict[str, list[dict]],
    policy_rows_by_key: dict[tuple[str, str, str], list[dict]],
) -> tuple[list[dict], list[str]]:
    rows = []
    available_methods = []
    for method in ["plain", "hybrid", "covariance"]:
        method_has_rows = False
        for qid, sweep_rows in records_by_query.items():
            metric_rows = policy_rows_by_key.get((domain, method, qid), [])
            for metric_row in metric_rows:
                selected = _nearest_lambda_row(sweep_rows, float(metric_row["chosen_lambda"]))
                rows.append(_policy_row_from_sweep(domain, qid, method, selected, float(selected["lambda"])))
                method_has_rows = True
        if method_has_rows:
            available_methods.append(method)
    return rows, available_methods


def _score_gap_rows_and_check(
    domain: str,
    records_by_query: dict[str, list[dict]],
    segments,
    segment_to_doc_id: dict[int, str],
    qrels: dict[str, dict[str, int]],
    embedder: MiniLMEmbedder,
    candidate_k: int,
    max_segments: int,
) -> tuple[list[dict], dict]:
    corpus_embs = np.asarray([segment.vector for segment in segments], dtype=np.float32)
    corpus_norms = np.linalg.norm(corpus_embs, axis=1)
    qids = list(records_by_query.keys())
    query_texts = [records_by_query[qid][0]["query"] for qid in qids]
    query_embs = np.asarray(embedder.embed(query_texts), dtype=np.float32)
    rows = []
    mismatches = []

    for qid, query, query_emb in zip(qids, query_texts, query_embs):
        query_emb = np.asarray(query_emb, dtype=np.float32).reshape(-1)
        query_norm = float(np.linalg.norm(query_emb))
        denom = np.maximum(corpus_norms * max(query_norm, EPS), EPS)
        scores = (corpus_embs @ query_emb) / denom

        indices_job4 = job4_score_gap_cutoff_indices(
            scores,
            min_k=1,
            max_k=max_segments,
            candidate_k=candidate_k,
        )
        indices_job5 = job5_score_gap_cutoff_indices(
            scores,
            min_k=1,
            max_k=max_segments,
            candidate_k=candidate_k,
        )
        docs_job4 = [
            segment_to_doc_id[int(segments[idx].start_idx)]
            for idx in indices_job4
            if int(segments[idx].start_idx) in segment_to_doc_id
        ]
        docs_job5 = [
            segment_to_doc_id[int(segments[idx].start_idx)]
            for idx in indices_job5
            if int(segments[idx].start_idx) in segment_to_doc_id
        ]
        if docs_job4 != docs_job5:
            mismatches.append({"question_id": qid, "job4_doc_ids": docs_job4, "job5_doc_ids": docs_job5})

        selected_segments = [segments[idx] for idx in indices_job4]
        ndcg = beir_ndcg(
            retrieved_segments=selected_segments,
            query_id=qid,
            qrels=qrels,
            segment_to_doc_id=segment_to_doc_id,
            k=10,
        )
        rows.append(
            {
                "domain": domain,
                "question_id": qid,
                "query": query,
                "method": "runtime_score_gap",
                "chosen_lambda": None,
                "runtime_candidate_k": int(min(candidate_k, len(scores))),
                **_metrics_from_retrieval(
                    ndcg_at_10=ndcg,
                    retrieved_doc_ids=docs_job4,
                    relevant_doc_ids=records_by_query[qid][0].get("relevant_doc_ids", []),
                    num_segments=len(selected_segments),
                ),
            }
        )

    return rows, {
        "score_gap_job4_job5_identical": len(mismatches) == 0,
        "score_gap_job4_job5_mismatch_count": len(mismatches),
        "score_gap_job4_job5_mismatches_preview": mismatches[:10],
    }


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


def _mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def _sanity_for_domain(domain: str, records_by_query: dict[str, list[dict]]) -> dict:
    tie_count = 0
    old_segments = []
    sparse_segments = []
    old_precision = []
    sparse_precision = []
    for rows in records_by_query.values():
        max_ndcg = max(float(row["ndcg_at_10"]) for row in rows)
        tied_rows = [row for row in rows if abs(float(row["ndcg_at_10"]) - max_ndcg) <= 1e-12]
        if len(tied_rows) > 1:
            tie_count += 1
        old = _old_first_tie_ndcg_oracle(rows)
        sparse = _sparse_ndcg_oracle(rows)
        old_segments.append(float(old.get("num_segments", 0)))
        sparse_segments.append(float(sparse.get("num_segments", 0)))
        old_precision.append(float(_metrics_from_sweep_row(old)["precision_returned"]))
        sparse_precision.append(float(_metrics_from_sweep_row(sparse)["precision_returned"]))

    n_queries = len(records_by_query)
    return {
        "domain": domain,
        "n_queries": int(n_queries),
        "max_ndcg_tie_query_pct": float(100.0 * tie_count / n_queries) if n_queries else 0.0,
        "old_first_tie_oracle_avg_num_segments": _mean(old_segments),
        "sparse_tie_oracle_avg_num_segments": _mean(sparse_segments),
        "old_first_tie_oracle_precision_returned": _mean(old_precision),
        "sparse_tie_oracle_precision_returned": _mean(sparse_precision),
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


def _bootstrap_summary(domain: str, rows_by_method: dict[str, list[dict]], n_samples: int, seed: int) -> dict:
    fixed_by_qid = defaultdict(list)
    for row in rows_by_method["fixed"]:
        fixed_by_qid[row["question_id"]].append(row)

    summary = {"domain": domain, "methods": {}, "diff_vs_fixed": {}}
    for method, rows in rows_by_method.items():
        method_by_qid = defaultdict(list)
        for row in rows:
            method_by_qid[row["question_id"]].append(row)
        summary["methods"][method] = {
            "n_queries": len(method_by_qid),
            "n_rows": len(rows),
        }
        for offset, metric in enumerate(TABLE_METRICS):
            per_query_values = [
                float(np.mean([float(row[metric]) for row in q_rows]))
                for q_rows in method_by_qid.values()
            ]
            summary["methods"][method][metric] = _bootstrap_mean_ci(per_query_values, n_samples, seed + offset)

            if method != "fixed":
                paired_method = []
                paired_fixed = []
                for qid, q_rows in method_by_qid.items():
                    if qid not in fixed_by_qid:
                        continue
                    paired_method.append(float(np.mean([float(row[metric]) for row in q_rows])))
                    paired_fixed.append(float(np.mean([float(row[metric]) for row in fixed_by_qid[qid]])))
                summary["diff_vs_fixed"].setdefault(method, {})[metric] = _paired_bootstrap_diff_ci(
                    paired_method,
                    paired_fixed,
                    n_samples,
                    seed + 100 + offset,
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
        for method in METHOD_ORDER:
            if method not in summary["methods"]:
                continue
            item = summary["methods"][method]
            row = {
                "domain": domain,
                "method": method,
                "n_queries": item["n_queries"],
                "nDCG@10": _format_ci(item["ndcg_at_10"]),
                "precision@returned": _format_ci(item["precision_returned"]),
                "Avg. returned segments": _format_ci(item["num_returned_segments"]),
                "Avg. unique docs": _format_ci(item["num_returned_docs"]),
            }
            if method == "fixed":
                row["Delta nDCG vs fixed"] = "--"
                row["Delta precision vs fixed"] = "--"
            else:
                deltas = summary["diff_vs_fixed"][method]
                row["Delta nDCG vs fixed"] = _format_delta(deltas["ndcg_at_10"])
                row["Delta precision vs fixed"] = _format_delta(deltas["precision_returned"])
            rows.append(row)
    return rows


def _evaluate_domain(
    domain: dict,
    fixed_lambda: float,
    policy_rows_by_key: dict[tuple[str, str, str], list[dict]],
    embedder: MiniLMEmbedder,
    segment_config: SegmentBuildConfig,
    candidate_k: int,
    max_segments: int,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> tuple[dict, list[dict], dict]:
    records = _read_json_rows(domain["records_path"])
    _assert_segment_config(records, segment_config, domain["records_path"])
    records_by_query = _group_by_query(records)

    sanity = _sanity_for_domain(domain["name"], records_by_query)
    rows_by_method: dict[str, list[dict]] = defaultdict(list)

    for qid, rows in records_by_query.items():
        fixed_row = _nearest_lambda_row(rows, fixed_lambda)
        rows_by_method["fixed"].append(
            _policy_row_from_sweep(domain["name"], qid, "fixed", fixed_row, float(fixed_row["lambda"]))
        )
        ndcg_oracle = _sparse_ndcg_oracle(rows)
        rows_by_method["ndcg_oracle"].append(
            _policy_row_from_sweep(
                domain["name"], qid, "ndcg_oracle", ndcg_oracle, float(ndcg_oracle["lambda"])
            )
        )
        precision_oracle = _precision_oracle(rows)
        rows_by_method["precision_oracle"].append(
            _policy_row_from_sweep(
                domain["name"], qid, "precision_oracle", precision_oracle, float(precision_oracle["lambda"])
            )
        )

    policy_rows, available_policy_methods = _policy_rows_from_existing_metrics(
        domain["name"],
        records_by_query,
        policy_rows_by_key,
    )
    for row in policy_rows:
        rows_by_method[row["method"]].append(row)

    data_path = resolve_beir_dataset_path(domain["data_path"], domain["name"], domain["split"])
    corpus, _, qrels = GenericDataLoader(data_path).load(split=domain["split"])
    segments, segment_to_doc_id = build_beir_segments(corpus, embedder, segment_config=segment_config)
    score_gap_rows, score_gap_check = _score_gap_rows_and_check(
        domain=domain["name"],
        records_by_query=records_by_query,
        segments=segments,
        segment_to_doc_id=segment_to_doc_id,
        qrels=qrels,
        embedder=embedder,
        candidate_k=candidate_k,
        max_segments=max_segments,
    )
    rows_by_method["runtime_score_gap"].extend(score_gap_rows)

    sanity.update(score_gap_check)
    sanity["available_existing_policy_methods"] = available_policy_methods

    all_rows = []
    run_row = {"domain": domain["name"], "n_queries": len(records_by_query)}
    for method in METHOD_ORDER:
        rows = rows_by_method.get(method, [])
        if not rows:
            continue
        all_rows.extend(rows)
        run_row[f"{method}_n_rows"] = len(rows)
        for metric in TABLE_METRICS:
            run_row[f"{method}_{metric}"] = float(np.mean([float(row[metric]) for row in rows]))

    bootstrap = _bootstrap_summary(
        domain=domain["name"],
        rows_by_method=dict(rows_by_method),
        n_samples=bootstrap_samples,
        seed=bootstrap_seed,
    )
    return run_row, all_rows, sanity, bootstrap


def main() -> None:
    configure_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument("--domains", default=",".join(DEFAULT_DOMAINS))
    parser.add_argument("--records-dir", default="outputs")
    parser.add_argument("--beir-data-dir", default="outputs/beir_data")
    parser.add_argument("--policy-query-metrics", default="")
    parser.add_argument("--fixed-lambda", type=float, default=0.1)
    parser.add_argument("--split", default="test")
    parser.add_argument("--embed-model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--output-dir", default="outputs/runtime_ablation/final_consistent_eval")
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
    fixed_lambda = float(args.fixed_lambda)
    if abs(fixed_lambda - 0.1) > 1e-12:
        LOGGER.warning("fixed_lambda is %s, expected paper setting is 0.1", fixed_lambda)

    segment_config = SegmentBuildConfig(
        strategy=args.segment_strategy,
        min_sentence_len=args.min_sentence_len,
        min_segment_size=args.min_segment_size,
        max_segment_size=args.max_segment_size,
        lookback_k=args.lookback_k,
    )
    domains = _parse_domains(args)
    explicit_policy_paths = [item.strip() for item in args.policy_query_metrics.split(",") if item.strip()]
    policy_paths = explicit_policy_paths or _auto_policy_metric_paths(args.records_dir)
    policy_rows_by_key = _load_policy_metric_rows(policy_paths)
    embedder = MiniLMEmbedder(args.embed_model)

    run_rows = []
    query_rows = []
    sanity_checks = {
        "fixed_lambda": fixed_lambda,
        "fixed_lambda_is_0_1": abs(fixed_lambda - 0.1) <= 1e-12,
        "policy_query_metric_paths": policy_paths,
        "domains": {},
        "stale_claim_flag": (
            "Any text/table claiming the nDCG oracle is permissive or has the lowest precision "
            "is stale after sparse tie-breaking."
        ),
    }
    bootstrap_summaries = []

    for domain in domains:
        LOGGER.info("running final consistent eval for %s", domain["name"])
        run_row, domain_query_rows, sanity, bootstrap = _evaluate_domain(
            domain=domain,
            fixed_lambda=fixed_lambda,
            policy_rows_by_key=policy_rows_by_key,
            embedder=embedder,
            segment_config=segment_config,
            candidate_k=args.runtime_candidate_k,
            max_segments=args.runtime_max_segments,
            bootstrap_samples=args.bootstrap_samples,
            bootstrap_seed=args.bootstrap_seed,
        )
        run_rows.append(run_row)
        query_rows.extend(domain_query_rows)
        sanity_checks["domains"][domain["name"]] = sanity
        bootstrap_summaries.append(bootstrap)
        print(
            f"{domain['name']}: n={sanity['n_queries']}, "
            f"max-nDCG ties={sanity['max_ndcg_tie_query_pct']:.1f}%, "
            f"old oracle segments={sanity['old_first_tie_oracle_avg_num_segments']:.2f}, "
            f"sparse oracle segments={sanity['sparse_tie_oracle_avg_num_segments']:.2f}, "
            f"old precision={sanity['old_first_tie_oracle_precision_returned']:.3f}, "
            f"sparse precision={sanity['sparse_tie_oracle_precision_returned']:.3f}, "
            f"score-gap job4/job5 identical={sanity['score_gap_job4_job5_identical']}"
        )
    print(f"fixed_lambda={fixed_lambda}")

    with open(os.path.join(args.output_dir, "final_consistent_runs.json"), "w", encoding="utf-8") as handle:
        json.dump(run_rows, handle, indent=2)
    _write_jsonl(os.path.join(args.output_dir, "final_consistent_query_metrics.jsonl"), query_rows)
    with open(
        os.path.join(args.output_dir, "final_consistent_bootstrap_summary.json"),
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(bootstrap_summaries, handle, indent=2)
    with open(os.path.join(args.output_dir, "sanity_checks.json"), "w", encoding="utf-8") as handle:
        json.dump(sanity_checks, handle, indent=2)

    table_rows = _paper_table_rows(bootstrap_summaries)
    _write_csv(os.path.join(args.output_dir, "final_consistent_paper_table.csv"), table_rows)
    _write_latex_table(os.path.join(args.output_dir, "final_consistent_paper_table.tex"), table_rows)

    save_run_metadata(
        os.path.join(args.output_dir, "metadata.json"),
        args,
        extra={
            "fixed_lambda": fixed_lambda,
            "segment_config": segment_config.to_dict(),
            "table_metrics": TABLE_METRICS,
            "oracle_tie_breaking": "nDCG ties break toward fewer admitted segments",
            "score_gap_ndcg_scorer": "shared.beir_scoring.beir_ndcg",
        },
    )
    LOGGER.info("saved final consistent outputs to %s", args.output_dir)


if __name__ == "__main__":
    main()

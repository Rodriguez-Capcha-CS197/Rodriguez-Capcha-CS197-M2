"""Week 9 scaling sweep over training set size."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

from configs.metadata import save_run_metadata
from scripts.reporting import export_scaling_latex

_JOBS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_JOBS_DIR, "..", ".."))


def _run_week9_eval(output_dir, n_train_limit, common_args):
    cmd = [
        sys.executable,
        os.path.join(_JOBS_DIR, "job3_week9_eval.py"),
        "--output-dir",
        output_dir,
        "--fineweb-records",
        common_args["fineweb_records"],
        "--marco-records",
        common_args["marco_records"],
        "--scifact-records",
        common_args["scifact_records"],
        "--fiqa-records",
        common_args["fiqa_records"],
        "--scifact-data-path",
        common_args["scifact_data_path"],
        "--fiqa-data-path",
        common_args["fiqa_data_path"],
        "--domains",
        common_args["domains"],
        "--split",
        common_args["split"],
        "--embed-model",
        common_args["embed_model"],
        "--seeds",
        common_args["seeds"],
        "--train-limit",
        str(n_train_limit),
        "--bootstrap-samples",
        str(common_args["bootstrap_samples"]),
        "--qualitative-examples-per-domain",
        "0",
        "--segment-strategy",
        common_args["segment_strategy"],
        "--min-sentence-len",
        str(common_args["min_sentence_len"]),
        "--min-segment-size",
        str(common_args["min_segment_size"]),
        "--max-segment-size",
        str(common_args["max_segment_size"]),
        "--lookback-k",
        str(common_args["lookback_k"]),
    ]
    env = {**os.environ, "PYTHONPATH": _REPO_ROOT}
    subprocess.run(cmd, check=True, cwd=_REPO_ROOT, env=env)


def _read_mean(path, key):
    with open(path, "r", encoding="utf-8") as f:
        rows = json.load(f)
    vals = [float(r[key]) for r in rows]
    return sum(vals) / len(vals) if vals else 0.0


def _domain_output_name(domain_spec):
    name = domain_spec.split(":", 1)[0]
    return name, name.replace("/", "_").replace(" ", "_")


def main():
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
        help="Comma-separated domains passed to job3_week9_eval.py.",
    )
    parser.add_argument("--split", default="test")
    parser.add_argument("--embed-model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--sizes", default="1000,5000,10000,25000,50000")
    parser.add_argument("--output-dir", default="outputs/week9/scaling")
    parser.add_argument("--bootstrap-samples", type=int, default=200)
    parser.add_argument("--segment-strategy", default="geometry_sentence")
    parser.add_argument("--min-sentence-len", type=int, default=20)
    parser.add_argument("--min-segment-size", type=int, default=2)
    parser.add_argument("--max-segment-size", type=int, default=15)
    parser.add_argument("--lookback-k", type=int, default=50)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    sizes = [int(x) for x in args.sizes.split(",") if x.strip()]

    common = {
        "fineweb_records": args.fineweb_records,
        "marco_records": args.marco_records,
        "scifact_records": args.scifact_records,
        "fiqa_records": args.fiqa_records,
        "scifact_data_path": args.scifact_data_path,
        "fiqa_data_path": args.fiqa_data_path,
        "domains": args.domains,
        "split": args.split,
        "embed_model": args.embed_model,
        "seeds": args.seeds,
        "bootstrap_samples": args.bootstrap_samples,
        "segment_strategy": args.segment_strategy,
        "min_sentence_len": args.min_sentence_len,
        "min_segment_size": args.min_segment_size,
        "max_segment_size": args.max_segment_size,
        "lookback_k": args.lookback_k,
    }

    summary = []
    for size in sizes:
        out = os.path.join(args.output_dir, f"n_{size}")
        os.makedirs(out, exist_ok=True)
        _run_week9_eval(out, size, common)

        row = {"n_train": size}
        for domain_spec in [item.strip() for item in args.domains.split(",") if item.strip()]:
            domain_name, output_name = _domain_output_name(domain_spec)
            result_path = os.path.join(out, f"{output_name}_results.json")
            row[f"{domain_name}_plain_ndcg@10"] = _read_mean(result_path, "plain")
            row[f"{domain_name}_plain_precision@returned"] = _read_mean(
                result_path,
                "plain_precision_returned",
            )
            row[f"{domain_name}_plain_p@5"] = _read_mean(result_path, "plain_precision_at_5")
            row[f"{domain_name}_plain_f1@returned"] = _read_mean(result_path, "plain_f1_returned")
            row[f"{domain_name}_plain_avg_returned"] = _read_mean(
                result_path,
                "plain_avg_num_returned_docs",
            )
        summary.append(row)
        print(row)

    with open(os.path.join(args.output_dir, "scaling_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    export_scaling_latex(
        os.path.join(args.output_dir, "scaling_summary.json"),
        os.path.join(args.output_dir, "scaling_summary_table.tex"),
    )
    save_run_metadata(
        os.path.join(args.output_dir, "scaling_metadata.json"),
        args,
        extra={"sizes": sizes},
    )


if __name__ == "__main__":
    main()

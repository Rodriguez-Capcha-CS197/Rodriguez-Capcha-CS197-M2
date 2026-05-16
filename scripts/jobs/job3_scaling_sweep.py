"""Week 9 scaling sweep over training set size."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys


def _run_week9_eval(output_dir, n_train_limit, common_args):
    cmd = [
        sys.executable,
        "jobs/job3_week9_eval.py",
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
        "--split",
        common_args["split"],
        "--embed-model",
        common_args["embed_model"],
        "--seeds",
        common_args["seeds"],
        "--train-limit",
        str(n_train_limit),
    ]
    subprocess.run(cmd, check=True)


def _read_mean(path, key):
    with open(path, "r", encoding="utf-8") as f:
        rows = json.load(f)
    vals = [float(r[key]) for r in rows]
    return sum(vals) / len(vals) if vals else 0.0


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
    parser.add_argument("--sizes", default="1000,5000,10000,25000,50000")
    parser.add_argument("--output-dir", default="outputs/week9/scaling")
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
        "split": args.split,
        "embed_model": args.embed_model,
        "seeds": args.seeds,
    }

    summary = []
    for size in sizes:
        out = os.path.join(args.output_dir, f"n_{size}")
        os.makedirs(out, exist_ok=True)
        _run_week9_eval(out, size, common)

        sf = _read_mean(os.path.join(out, "scifact_results.json"), "plain")
        fq = _read_mean(os.path.join(out, "fiqa_results.json"), "plain")
        row = {"n_train": size, "scifact_plain_ndcg@10": sf, "fiqa_plain_ndcg@10": fq}
        summary.append(row)
        print(row)

    with open(os.path.join(args.output_dir, "scaling_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()

"""Paper table and visualization exports."""

from __future__ import annotations

import json
import os
from collections import defaultdict

import numpy as np
import pandas as pd


def _ensure_parent(path: str) -> None:
    output_dir = os.path.dirname(path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)


def read_json_rows(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    if isinstance(data, list):
        return data
    return [data]


def export_latex_table(df: pd.DataFrame, path: str, *, index: bool = False, float_format: str = "%.3f") -> None:
    _ensure_parent(path)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(df.to_latex(index=index, float_format=float_format))


def export_scaling_latex(summary_path: str, output_path: str) -> None:
    rows = read_json_rows(summary_path)
    df = pd.DataFrame(rows).sort_values("n_train")
    export_latex_table(df, output_path, index=False)


def _mean_std(values: list[float]) -> str:
    if not values:
        return "0.000 ± 0.000"
    arr = np.asarray(values, dtype=np.float32)
    return f"{float(np.mean(arr)):.3f} ± {float(np.std(arr)):.3f}"


def export_cross_domain_latex(scifact_results_path: str, fiqa_results_path: str, output_path: str) -> None:
    rows = []
    for domain, path in [("SciFact", scifact_results_path), ("FiQA", fiqa_results_path)]:
        runs = read_json_rows(path)
        methods = [key for key in ["oracle", "fixed", "plain", "hybrid", "covariance"] if key in runs[0]]
        for method in methods:
            rows.append(
                {
                    "domain": domain,
                    "method": method,
                    "nDCG@10": _mean_std([float(run[method]) for run in runs]),
                }
            )
    export_latex_table(pd.DataFrame(rows), output_path, index=False, float_format="%s")


def export_cross_domain_precision_latex(scifact_results_path: str, fiqa_results_path: str, output_path: str) -> None:
    rows = []
    for domain, path in [("SciFact", scifact_results_path), ("FiQA", fiqa_results_path)]:
        runs = read_json_rows(path)
        methods = [key for key in ["oracle", "fixed", "plain", "hybrid", "covariance"] if key in runs[0]]
        for method in methods:
            rows.append(
                {
                    "domain": domain,
                    "method": method,
                    "precision@returned": _mean_std(
                        [float(run[f"{method}_precision_returned"]) for run in runs]
                    ),
                    "P@5": _mean_std([float(run[f"{method}_precision_at_5"]) for run in runs]),
                    "Recall@5": _mean_std([float(run[f"{method}_recall_at_5"]) for run in runs]),
                }
            )
    export_latex_table(pd.DataFrame(rows), output_path, index=False, float_format="%s")


def export_kshot_latex(result_path: str, output_path: str) -> None:
    rows = read_json_rows(result_path)
    columns = [
        "k_shot",
        "mlp_zero_shot_held_out_ndcg_at_10",
        "mlp_k_shot_ft_ndcg_at_10",
        "oracle_held_out_ndcg_at_10",
        "recovery_pct",
        "held_out_eval_queries",
    ]
    df = pd.DataFrame([{key: row.get(key) for key in columns} for row in rows])
    export_latex_table(df, output_path, index=False)


def _optimal_rows(records: list[dict]) -> list[dict]:
    return [row for row in records if row.get("is_optimal")]


def plot_lambda_distribution_shift(fineweb_records_path: str, marco_records_path: str, output_png: str) -> None:
    import matplotlib.pyplot as plt

    fineweb = _optimal_rows(read_json_rows(fineweb_records_path))
    marco = _optimal_rows(read_json_rows(marco_records_path))
    if not fineweb or not marco:
        raise ValueError("Need optimal rows in both FineWeb and MS MARCO records for lambda shift plot.")

    fineweb_lambdas = [float(row["lambda"]) for row in fineweb]
    marco_lambdas = [float(row["lambda"]) for row in marco]
    bins = sorted(set(fineweb_lambdas + marco_lambdas))
    x = np.arange(len(bins))
    width = 0.4

    def counts(values):
        total = max(1, len(values))
        return [sum(1 for value in values if value == lam) / total for lam in bins]

    plt.figure(figsize=(8, 4))
    plt.bar(x - width / 2, counts(fineweb_lambdas), width=width, label="FineWeb synthetic queries")
    plt.bar(x + width / 2, counts(marco_lambdas), width=width, label="MS MARCO queries")
    plt.xticks(x, [str(lam) for lam in bins], rotation=30, ha="right")
    plt.ylabel("Fraction of optimal queries")
    plt.xlabel("Optimal lambda")
    plt.legend()
    plt.tight_layout()
    _ensure_parent(output_png)
    plt.savefig(output_png, dpi=200)
    plt.savefig(os.path.splitext(output_png)[0] + ".pdf")
    plt.close()


def plot_feature_ablation(scifact_runs: list[dict], fiqa_runs: list[dict], output_png: str) -> None:
    import matplotlib.pyplot as plt

    methods = ["plain", "hybrid", "covariance"]
    domains = [("SciFact", scifact_runs), ("FiQA", fiqa_runs)]
    x = np.arange(len(methods))
    width = 0.35

    plt.figure(figsize=(7, 4))
    for offset, (domain, runs) in zip([-width / 2, width / 2], domains):
        means = [float(np.mean([run[method] for run in runs])) for method in methods]
        plt.bar(x + offset, means, width=width, label=domain)
    plt.xticks(x, methods)
    plt.ylabel("nDCG@10")
    plt.legend()
    plt.tight_layout()
    _ensure_parent(output_png)
    plt.savefig(output_png, dpi=200)
    plt.savefig(os.path.splitext(output_png)[0] + ".pdf")
    plt.close()


def plot_tsne_lambda(records: list[dict], output_png: str, seed: int = 0, max_points: int = 3000) -> None:
    import matplotlib.pyplot as plt

    optimal = [row for row in _optimal_rows(records) if "plain_features" in row]
    if len(optimal) < 3:
        raise ValueError("Need at least three optimal rows with plain_features for t-SNE.")
    if len(optimal) > max_points:
        rng = np.random.default_rng(seed)
        indices = rng.choice(len(optimal), size=max_points, replace=False)
        optimal = [optimal[int(index)] for index in indices]

    try:
        from sklearn.manifold import TSNE
    except ImportError as exc:
        raise RuntimeError("scikit-learn is required for t-SNE plotting.") from exc

    X = np.asarray([row["plain_features"] for row in optimal], dtype=np.float32)
    lambdas = np.asarray([float(row["lambda"]) for row in optimal], dtype=np.float32)
    perplexity = min(30, max(2, (len(optimal) - 1) // 3))
    coords = TSNE(n_components=2, perplexity=perplexity, init="pca", random_state=seed).fit_transform(X)

    plt.figure(figsize=(6, 5))
    points = plt.scatter(coords[:, 0], coords[:, 1], c=lambdas, cmap="viridis", s=9, alpha=0.8)
    plt.colorbar(points, label="Optimal lambda")
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    _ensure_parent(output_png)
    plt.savefig(output_png, dpi=220)
    plt.savefig(os.path.splitext(output_png)[0] + ".pdf")
    plt.close()


def summarize_feature_presence(records: list[dict]) -> dict[str, int]:
    counts = defaultdict(int)
    for row in _optimal_rows(records):
        for key in ["plain_features", "density_features", "hybrid_features", "covariance_features"]:
            if key in row:
                counts[key] += 1
    return dict(counts)

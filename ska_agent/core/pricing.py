"""
Stage II: Pricing engine and lambda-selection utilities.

Given a query and a set of Segments from the geometry learner
(`core/geometry.py`), this module selects the most informative segments
using a pricing-guided greedy algorithm.

The master objective being minimized is:

    Phi(V) = phi(V) + lambda * |V| + eta * R(V)

where phi(V) is reconstruction error, lambda * |V| is a sparsity
penalty, and R(V) penalizes redundant selections. The reduced cost for
adding segment j is:

    c_bar(j) = lambda + eta * delta_R - delta_phi

If c_bar(j) < 0, adding segment j improves the objective, so the engine
greedily selects the best candidate and updates the residual by
orthogonal projection.

The second half of the module provides a clean, runnable replacement for
the experimental notebook snippets that were previously pasted here:
coverage metrics, lambda sweeping, dataset construction, a lightweight
lambda regressor, and optional plotting helpers.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

if __package__ in {None, ""}:
    package_root = Path(__file__).resolve().parents[2]
    if str(package_root) not in sys.path:
        sys.path.insert(0, str(package_root))
    from ska_agent.core.structures import RetrievalResult, Segment
    from ska_agent.utils.math_utils import MathUtils
else:
    from ..core.structures import RetrievalResult, Segment
    from ..utils.math_utils import MathUtils


DEFAULT_LAMBDA_GRID = np.logspace(-3, 1, 40)
DEFAULT_ALPHA = 0.01
DEFAULT_ETA = 0.0
DEFAULT_MAX_SEGMENTS = 5
EPS = 1e-10


def _ensure_1d(array: np.ndarray) -> np.ndarray:
    """Return a flattened float64 vector."""
    arr = np.asarray(array, dtype=np.float64)
    return arr.reshape(-1)


def _ensure_2d_rows(array: Sequence[np.ndarray]) -> np.ndarray:
    """Normalize a sequence of vectors into a 2D row-major array."""
    arr = np.asarray(array, dtype=np.float64)
    if arr.size == 0:
        return np.empty((0, 0), dtype=np.float64)
    if arr.ndim == 1:
        return arr.reshape(1, -1)
    return arr


def _normalize(vector: np.ndarray) -> np.ndarray:
    """L2-normalize a vector, leaving all-zero vectors unchanged."""
    norm = np.linalg.norm(vector)
    return vector if norm < EPS else vector / norm


class PricingEngine:
    """
    Pricing-guided discrete selection for retrieval.

    The information gain is computed via the Schur complement formula:
        delta_phi = (residual · segment)^2 / ||segment||^2

    After selection, the residual is updated by orthogonal projection.
    """

    def __init__(
        self,
        segments: List[Segment],
        embed_fn: Callable[[str], np.ndarray],
        lambda_sparsity: float = 0.05,
        lambda_fn: Optional[Callable[[str], float]] = None,
        eta_redundancy: float = 0.0,
        max_segments: int = 5,
        stopping_threshold: float = 1e-6,
    ):
        self.segments = segments
        self.embed_fn = embed_fn
        self.lambda_sparsity = float(lambda_sparsity)
        self.lambda_fn = lambda_fn
        self.eta_redundancy = float(eta_redundancy)
        self.max_segments = int(max_segments)
        self.stopping_threshold = float(stopping_threshold)

        if segments:
            self.U = np.asarray([_ensure_1d(s.vector) for s in segments], dtype=np.float64)
        else:
            self.U = np.empty((0, 0), dtype=np.float64)
        self.norms_sq = np.sum(self.U ** 2, axis=1) if len(self.U) else np.empty(0, dtype=np.float64)

    def compute_information_gain(self, residual: np.ndarray, j: int) -> float:
        """Schur complement: delta_phi = (residual · segment)^2 / ||segment||^2."""
        if self.norms_sq[j] < EPS:
            return 0.0
        inner = float(np.dot(self.U[j], residual))
        return (inner ** 2) / float(self.norms_sq[j])

    def compute_redundancy_penalty(self, j: int, selected: List[int]) -> float:
        """Use max cosine similarity to already-selected segments."""
        if not selected or self.eta_redundancy == 0.0:
            return 0.0

        u_j = self.U[j]
        norm_j = np.linalg.norm(u_j)
        if norm_j < EPS:
            return 0.0

        max_sim = 0.0
        for idx in selected:
            u_i = self.U[idx]
            denom = norm_j * np.linalg.norm(u_i)
            if denom < EPS:
                continue
            max_sim = max(max_sim, float(np.dot(u_j, u_i) / denom))
        return max_sim

    def compute_reduced_cost(
        self,
        residual: np.ndarray,
        j: int,
        selected: List[int],
        lambda_value: float,
    ) -> Tuple[float, float]:
        """Return (reduced_cost, information_gain)."""
        delta_phi = self.compute_information_gain(residual, j)
        delta_R = self.compute_redundancy_penalty(j, selected)
        reduced_cost = lambda_value + self.eta_redundancy * delta_R - delta_phi
        return reduced_cost, delta_phi

    def update_residual(self, residual: np.ndarray, j: int) -> np.ndarray:
        """Remove the component explained by segment j."""
        return MathUtils.orthogonal_projection(residual, self.U[j])

    def retrieve(self, query: str, verbose: bool = True) -> RetrievalResult:
        """Run pricing-guided retrieval with monotone descent."""
        if not self.segments:
            return RetrievalResult(segments=[], reduced_costs=[], total_segments_considered=0)

        query_lambda = float(self.lambda_fn(query)) if self.lambda_fn is not None else self.lambda_sparsity
        query_emb = _ensure_1d(self.embed_fn(query))
        residual = _normalize(query_emb)

        selected_indices: List[int] = []
        reduced_costs: List[float] = []
        visited = np.zeros(len(self.segments), dtype=bool)

        if verbose:
            print(f"\nQuery: '{query}'")
            print(f" Retrieval (lambda={query_lambda:.3f}):")

        for iteration in range(self.max_segments):
            best_j, best_rc, best_ig = -1, float("inf"), 0.0

            for j in range(len(self.segments)):
                if visited[j]:
                    continue
                rc, ig = self.compute_reduced_cost(residual, j, selected_indices, query_lambda)
                if rc < best_rc:
                    best_rc, best_j, best_ig = rc, j, ig

            if best_j == -1:
                break

            if verbose:
                print(f" Iter {iteration + 1}: reduced_cost={best_rc:.4f}, info_gain={best_ig:.4f}")

            if best_rc >= -self.stopping_threshold:
                if verbose:
                    print(" Stopping: reduced_cost >= 0")
                break

            selected_indices.append(best_j)
            reduced_costs.append(best_rc)
            visited[best_j] = True

            if verbose:
                preview = self.segments[best_j].text[:60]
                suffix = "..." if len(self.segments[best_j].text) > 60 else ""
                print(f" Selected segment {best_j}: '{preview}{suffix}'")

            residual = self.update_residual(residual, best_j)
            if np.linalg.norm(residual) < 1e-6:
                if verbose:
                    print(" Stopping: residual depleted")
                break

        if verbose:
            print(f" Retrieved {len(selected_indices)} segments")

        return RetrievalResult(
            segments=[self.segments[i] for i in selected_indices],
            reduced_costs=reduced_costs,
            total_segments_considered=len(self.segments),
        )


def run_pricing_engine(
    query: str,
    segments: List[Segment],
    embed_fn: Callable[[str], np.ndarray],
    lam: float,
    eta: float = DEFAULT_ETA,
    max_segments: int = DEFAULT_MAX_SEGMENTS,
    stopping_threshold: float = 1e-6,
    verbose: bool = False,
) -> RetrievalResult:
    """Convenience wrapper to run the pricing engine for one query."""
    engine = PricingEngine(
        segments=segments,
        embed_fn=embed_fn,
        lambda_sparsity=lam,
        eta_redundancy=eta,
        max_segments=max_segments,
        stopping_threshold=stopping_threshold,
    )
    return engine.retrieve(query, verbose=verbose)


def get_selected_embeddings(result: RetrievalResult) -> np.ndarray:
    """Extract selected segment embeddings as a 2D array."""
    if not result.segments:
        return np.empty((0, 0), dtype=np.float64)
    return np.asarray([_ensure_1d(segment.vector) for segment in result.segments], dtype=np.float64)


def segment_document(path: str | Path) -> List[Segment]:
    """
    Segment a plain-text document by paragraph with a sentence fallback.

    This helper intentionally stays lightweight so the module remains
    runnable without the full geometry pipeline.
    """
    text = Path(path).read_text(encoding="utf-8")
    chunks = [chunk.strip() for chunk in re.split(r"\n\s*\n", text) if chunk.strip()]
    if not chunks:
        chunks = [line.strip() for line in text.splitlines() if line.strip()]

    segments: List[Segment] = []
    cursor = 0
    for chunk in chunks:
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", chunk) if s.strip()]
        length = max(1, len(sentences))
        segments.append(
            Segment(
                text=chunk,
                vector=np.zeros(0, dtype=np.float64),
                start_idx=cursor,
                end_idx=cursor + length,
                sentences=sentences,
                internal_cost=0.0,
            )
        )
        cursor += length
    return segments


def embed_segments(
    segs: Sequence[Segment],
    embed_fn: Callable[[str], np.ndarray],
    update_segments: bool = True,
) -> np.ndarray:
    """Embed segment text and optionally store the vectors back on the segments."""
    embeddings = np.asarray([_ensure_1d(embed_fn(seg.text)) for seg in segs], dtype=np.float64)
    if update_segments:
        for seg, emb in zip(segs, embeddings):
            seg.vector = emb
    return embeddings


def select_segments(
    query_emb: np.ndarray,
    seg_embs: np.ndarray,
    lam: float,
    eta: float = DEFAULT_ETA,
    max_segments: int = DEFAULT_MAX_SEGMENTS,
    stopping_threshold: float = 1e-6,
) -> List[int]:
    """
    Greedy reduced-cost selection directly on embeddings.

    This mirrors `PricingEngine.retrieve` but returns only indices.
    """
    query_emb = _ensure_1d(query_emb)
    seg_embs = _ensure_2d_rows(seg_embs)
    if seg_embs.size == 0:
        return []

    residual = _normalize(query_emb)
    norms_sq = np.sum(seg_embs ** 2, axis=1)
    selected: List[int] = []
    visited = np.zeros(seg_embs.shape[0], dtype=bool)

    for _ in range(max_segments):
        best_idx = -1
        best_rc = float("inf")

        for j in range(seg_embs.shape[0]):
            if visited[j]:
                continue

            if norms_sq[j] < EPS:
                info_gain = 0.0
            else:
                inner = float(np.dot(seg_embs[j], residual))
                info_gain = (inner ** 2) / float(norms_sq[j])

            redundancy = 0.0
            if eta != 0.0 and selected:
                norm_j = np.linalg.norm(seg_embs[j])
                for idx in selected:
                    denom = norm_j * np.linalg.norm(seg_embs[idx])
                    if denom < EPS:
                        continue
                    redundancy = max(redundancy, float(np.dot(seg_embs[j], seg_embs[idx]) / denom))

            rc = float(lam) + float(eta) * redundancy - info_gain
            if rc < best_rc:
                best_rc = rc
                best_idx = j

        if best_idx == -1 or best_rc >= -stopping_threshold:
            break

        selected.append(best_idx)
        visited[best_idx] = True
        residual = MathUtils.orthogonal_projection(residual, seg_embs[best_idx])
        if np.linalg.norm(residual) < 1e-6:
            break

    return selected


def coverage_signal(query_emb: np.ndarray, selected_seg_embs: np.ndarray) -> float:
    """
    Measure how much of the query direction is explained by the selected segments.
    """
    query_emb = _ensure_1d(query_emb)
    selected_seg_embs = _ensure_2d_rows(selected_seg_embs)
    if query_emb.size == 0 or selected_seg_embs.size == 0:
        return 0.0

    q_norm = np.linalg.norm(query_emb)
    if q_norm < EPS:
        return 0.0

    basis, _ = np.linalg.qr(selected_seg_embs.T, mode="reduced")
    if basis.size == 0:
        return 0.0

    projection = basis @ (basis.T @ query_emb)
    return float(np.linalg.norm(projection) / q_norm)


def composite_signal(
    query_emb: np.ndarray,
    selected_seg_embs: np.ndarray,
    alpha: float = DEFAULT_ALPHA,
    selected_seg_costs: Optional[Sequence[float]] = None,
) -> float:
    """
    Combine coverage with a simple selection-cost penalty.

    If explicit segment costs are not provided, each selected segment has
    unit cost.
    """
    selected_seg_embs = _ensure_2d_rows(selected_seg_embs)
    cov = coverage_signal(query_emb, selected_seg_embs)
    if selected_seg_embs.size == 0:
        return cov

    if selected_seg_costs is None:
        cost = float(selected_seg_embs.shape[0])
    else:
        cost = float(np.sum(np.asarray(selected_seg_costs, dtype=np.float64)))
    return cov - float(alpha) * cost


def best_lambda_for_query(
    query: str,
    segments: Sequence[Segment],
    embed_fn: Callable[[str], np.ndarray],
    lambda_grid: Optional[Sequence[float]] = None,
    alpha: float = DEFAULT_ALPHA,
    eta: float = DEFAULT_ETA,
    max_segments: int = DEFAULT_MAX_SEGMENTS,
) -> float:
    """Sweep lambda values and return the best one for a single query."""
    grid = np.asarray(lambda_grid if lambda_grid is not None else DEFAULT_LAMBDA_GRID, dtype=np.float64)
    query_emb = _ensure_1d(embed_fn(query))

    best_lambda = float(grid[0])
    best_score = -float("inf")
    for lam in grid:
        result = run_pricing_engine(
            query=query,
            segments=list(segments),
            embed_fn=embed_fn,
            lam=float(lam),
            eta=eta,
            max_segments=max_segments,
            verbose=False,
        )
        selected_embs = get_selected_embeddings(result)
        score = composite_signal(query_emb, selected_embs, alpha=alpha)
        if score > best_score:
            best_score = score
            best_lambda = float(lam)
    return best_lambda


def build_dataset(
    queries: Sequence[str],
    segments: Sequence[Segment],
    embed_fn: Callable[[str], np.ndarray],
    lambda_grid: Optional[Sequence[float]] = None,
    alpha: float = DEFAULT_ALPHA,
    eta: float = DEFAULT_ETA,
    max_segments: int = DEFAULT_MAX_SEGMENTS,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build query embeddings X and per-query best lambdas y."""
    X = np.asarray([_ensure_1d(embed_fn(q)) for q in queries], dtype=np.float64)
    y = np.asarray(
        [
            best_lambda_for_query(
                q,
                segments=segments,
                embed_fn=embed_fn,
                lambda_grid=lambda_grid,
                alpha=alpha,
                eta=eta,
                max_segments=max_segments,
            )
            for q in queries
        ],
        dtype=np.float64,
    )
    return X, y


class LambdaMLP:
    """
    Lightweight lambda regressor with a numpy training loop.

    The model predicts log(lambda) using a one-hidden-layer MLP and maps
    back to positive lambda values with `exp`.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 64, seed: int = 0):
        rng = np.random.default_rng(seed)
        scale1 = 1.0 / np.sqrt(max(1, input_dim))
        scale2 = 1.0 / np.sqrt(max(1, hidden_dim))

        self.W1 = rng.normal(0.0, scale1, size=(input_dim, hidden_dim))
        self.b1 = np.zeros(hidden_dim, dtype=np.float64)
        self.W2 = rng.normal(0.0, scale2, size=(hidden_dim, 1))
        self.b2 = np.zeros(1, dtype=np.float64)

    @staticmethod
    def _relu(x: np.ndarray) -> np.ndarray:
        return np.maximum(x, 0.0)

    @staticmethod
    def _relu_grad(x: np.ndarray) -> np.ndarray:
        return (x > 0.0).astype(np.float64)

    def predict_log(self, X: np.ndarray) -> np.ndarray:
        X = _ensure_2d_rows(X)
        hidden_pre = X @ self.W1 + self.b1
        hidden = self._relu(hidden_pre)
        return (hidden @ self.W2 + self.b2).reshape(-1)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.exp(self.predict_log(X))

    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 200, lr: float = 1e-2) -> "LambdaMLP":
        X = _ensure_2d_rows(X)
        y = _ensure_1d(y)
        y_log = np.log(np.clip(y, EPS, None))

        for _ in range(epochs):
            hidden_pre = X @ self.W1 + self.b1
            hidden = self._relu(hidden_pre)
            pred_log = (hidden @ self.W2 + self.b2).reshape(-1)

            error = pred_log - y_log
            grad_out = (2.0 / len(X)) * error.reshape(-1, 1)

            grad_W2 = hidden.T @ grad_out
            grad_b2 = grad_out.sum(axis=0)

            grad_hidden = (grad_out @ self.W2.T) * self._relu_grad(hidden_pre)
            grad_W1 = X.T @ grad_hidden
            grad_b1 = grad_hidden.sum(axis=0)

            self.W2 -= lr * grad_W2
            self.b2 -= lr * grad_b2
            self.W1 -= lr * grad_W1
            self.b1 -= lr * grad_b1

        return self


def train(
    X: np.ndarray,
    y: np.ndarray,
    epochs: int = 200,
    lr: float = 1e-2,
    hidden_dim: int = 64,
    seed: int = 0,
) -> LambdaMLP:
    """Train a lightweight lambda regressor."""
    X = _ensure_2d_rows(X)
    model = LambdaMLP(input_dim=X.shape[1], hidden_dim=hidden_dim, seed=seed)
    return model.fit(X, y, epochs=epochs, lr=lr)


def evaluate(
    queries: Sequence[str],
    segments: Sequence[Segment],
    embed_fn: Callable[[str], np.ndarray],
    lambda_fn: Callable[[str], float],
    alpha: float = DEFAULT_ALPHA,
    eta: float = DEFAULT_ETA,
    max_segments: int = DEFAULT_MAX_SEGMENTS,
) -> Any:
    """
    Evaluate a lambda policy over a set of queries.

    Returns a pandas DataFrame when pandas is available, otherwise a list
    of row dictionaries.
    """
    rows: List[Dict[str, Any]] = []
    for query in queries:
        lam = float(lambda_fn(query))
        result = run_pricing_engine(
            query=query,
            segments=list(segments),
            embed_fn=embed_fn,
            lam=lam,
            eta=eta,
            max_segments=max_segments,
            verbose=False,
        )
        query_emb = _ensure_1d(embed_fn(query))
        selected_embs = get_selected_embeddings(result)
        rows.append(
            {
                "query": query,
                "lambda": lam,
                "coverage": coverage_signal(query_emb, selected_embs),
                "num_segs": len(result.segments),
                "composite": composite_signal(query_emb, selected_embs, alpha=alpha),
            }
        )

    try:
        import pandas as pd

        return pd.DataFrame(rows)
    except ImportError:
        return rows


def plot(results_by_strategy: Dict[str, Any]) -> None:
    """
    Plot mean coverage and composite score for each strategy.

    Accepts either pandas DataFrames or lists of row dictionaries.
    """
    import matplotlib.pyplot as plt

    names: List[str] = []
    coverage_means: List[float] = []
    composite_means: List[float] = []

    for name, results in results_by_strategy.items():
        if hasattr(results, "to_dict"):
            records = results.to_dict(orient="records")
        else:
            records = list(results)

        if not records:
            continue

        names.append(name)
        coverage_means.append(float(np.mean([row["coverage"] for row in records])))
        composite_means.append(float(np.mean([row["composite"] for row in records])))

    x = np.arange(len(names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width / 2, coverage_means, width, label="coverage")
    ax.bar(x + width / 2, composite_means, width, label="composite")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_ylabel("score")
    ax.set_title("Pricing strategy comparison")
    ax.legend()
    fig.tight_layout()


__all__ = [
    "DEFAULT_ALPHA",
    "DEFAULT_ETA",
    "DEFAULT_LAMBDA_GRID",
    "DEFAULT_MAX_SEGMENTS",
    "LambdaMLP",
    "PricingEngine",
    "best_lambda_for_query",
    "build_dataset",
    "composite_signal",
    "coverage_signal",
    "embed_segments",
    "evaluate",
    "get_selected_embeddings",
    "plot",
    "run_pricing_engine",
    "segment_document",
    "select_segments",
    "train",
]


if __name__ == "__main__":
    demo_segments = [
        Segment(
            text="Alpha systems improve retrieval quality for structured queries.",
            vector=np.array([1.0, 0.1], dtype=np.float64),
            start_idx=0,
            end_idx=1,
        ),
        Segment(
            text="Beta planning helps coordinate multi-step reasoning tasks.",
            vector=np.array([0.7, 0.4], dtype=np.float64),
            start_idx=1,
            end_idx=2,
        ),
        Segment(
            text="Delta analysis is strongest when the query is about evaluation.",
            vector=np.array([0.0, 1.0], dtype=np.float64),
            start_idx=2,
            end_idx=3,
        ),
    ]

    def demo_embed_fn(text: str) -> np.ndarray:
        lower = text.lower()
        return np.array(
            [
                1.0 if any(token in lower for token in ("alpha", "retrieval", "structured")) else 0.0,
                1.0 if any(token in lower for token in ("delta", "evaluation", "analysis")) else 0.0,
            ],
            dtype=np.float64,
        )

    engine = PricingEngine(
        segments=demo_segments,
        embed_fn=demo_embed_fn,
        lambda_sparsity=0.2,
        max_segments=2,
    )

    print("PricingEngine smoke test")
    result = engine.retrieve("structured retrieval evaluation", verbose=False)
    print(f"Selected {len(result.segments)} segment(s)")
    for idx, (segment, cost) in enumerate(zip(result.segments, result.reduced_costs), start=1):
        print(f"{idx}. reduced_cost={cost:.4f} text={segment.text}")

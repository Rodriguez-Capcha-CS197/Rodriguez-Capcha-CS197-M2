"""Lambda sweep over a question set against a fixed segment corpus."""

import numpy as np
from ska_agent.core.pricing import PricingEngine
from .constants import LAMBDA_GRID, ETA_REDUNDANCY, MAX_SEGMENTS, TIE_EPS
from .scoring import complexity_aligned_lambda_score, get_selected_indices, ensure_1d


def generate_lambda_sweep_records(segments, questions, relevant_by_qid, embed_fn, lambda_grid=None):
    """Sweep lambda over the grid for each question. Returns list of records.

    Args:
        segments: list of Segment objects (already embedded).
        questions: list of OfficeQAQuestion objects.
        relevant_by_qid: dict qid -> list of gold segment indices.
        embed_fn: callable (text) -> 1D vector.
        lambda_grid: list of lambdas to sweep. Defaults to LAMBDA_GRID constant.
    """
    if lambda_grid is None:
        lambda_grid = LAMBDA_GRID

    records = []
    query_embedding_cache = {}
    for question in questions:
        query_embedding_cache[question.question] = ensure_1d(embed_fn(question.question))

    def cached_embed_fn(text):
        if text in query_embedding_cache:
            return query_embedding_cache[text]
        return ensure_1d(embed_fn(text))

    engines = {
        lam: PricingEngine(
            segments=segments, embed_fn=cached_embed_fn,
            lambda_sparsity=lam, eta_redundancy=ETA_REDUNDANCY,
            max_segments=MAX_SEGMENTS,
        )
        for lam in lambda_grid
    }

    for idx, question in enumerate(questions, start=1):
        qid = question.question_id
        query_text = question.question
        relevant_indices = relevant_by_qid[qid]
        records_for_this_query = []

        best_score = -float("inf")
        best_index = 0
        best_num_segments = None
        best_lambda = None

        for lam in lambda_grid:
            engine = engines[lam]
            result = engine.retrieve(query_text, verbose=False)
            selected_indices = get_selected_indices(result)
            selected_texts = [seg.text for seg in result.segments]

            final_score, metrics = complexity_aligned_lambda_score(
                selected_indices=selected_indices,
                relevant_indices=relevant_indices,
                mode=question.question_type,
            )

            if result.reduced_costs is not None and len(result.reduced_costs) > 0:
                total_reduced_cost = float(np.sum(result.reduced_costs))
            else:
                total_reduced_cost = 0.0

            row = {
                "question_id": qid, "query": query_text,
                "mode": question.question_type, "difficulty": question.difficulty,
                "gold_answer": question.answer, "lambda": float(lam),
                "num_segments": int(len(result.segments)),
                "total_reduced_cost": total_reduced_cost,
                **{k: v for k, v in metrics.items()},
                "retrieval_precision": metrics["precision"],
                "retrieval_recall": metrics["recall"],
                "retrieval_f1": metrics["f1"],
                "relevant_segment_indices": [int(x) for x in relevant_indices],
                "selected_segment_indices": selected_indices,
                "selected_segment_texts": selected_texts,
                "final_lambda_score": float(final_score),
                "is_optimal": False,
            }
            records_for_this_query.append(row)

            num_segments = len(result.segments)
            if final_score > best_score + TIE_EPS:
                best_score = final_score
                best_index = len(records_for_this_query) - 1
                best_num_segments = num_segments
                best_lambda = lam
            elif abs(final_score - best_score) <= TIE_EPS:
                if best_num_segments is None or num_segments < best_num_segments:
                    best_score = final_score
                    best_index = len(records_for_this_query) - 1
                    best_num_segments = num_segments
                    best_lambda = lam
                elif num_segments == best_num_segments and best_lambda is not None and lam > best_lambda:
                    best_score = final_score
                    best_index = len(records_for_this_query) - 1
                    best_num_segments = num_segments
                    best_lambda = lam

        records_for_this_query[best_index]["is_optimal"] = True
        records.extend(records_for_this_query)

        if idx % 50 == 0:
            print(f"  processed {idx}/{len(questions)} questions...")

    return records
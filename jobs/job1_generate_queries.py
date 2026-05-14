"""Job 1: Generate FineWeb-Edu synthetic queries."""

from __future__ import annotations

import argparse
import json
import os
from typing import List

from shared.fineweb_loader import load_fineweb_sample


PROMPT_TEMPLATE = """Generate 2 search queries a reader might use to find this passage.
Return only the queries, one per line, no numbering.

Passage:
{passage_text}
"""


def _template_queries(passage_text: str, n_queries: int) -> List[str]:
    """CPU fallback query generator when Qwen is unavailable."""
    stripped = " ".join(passage_text.strip().split())
    preview = stripped[:180]
    queries = [
        f"What is this passage about: {preview}?",
        f"Key ideas in: {preview}",
        f"Summary search: {preview}",
    ]
    return queries[:n_queries]


def _qwen_queries(passage_text: str, model_id: str, n_queries: int, temperature: float) -> List[str]:
    from transformers import pipeline
    import torch

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    pipe = pipeline("text-generation", model=model_id, dtype=dtype, device_map="auto")

    prompt = PROMPT_TEMPLATE.format(passage_text=passage_text)
    messages = [{"role": "user", "content": prompt}]
    outputs = pipe(messages, max_new_tokens=120, do_sample=True, temperature=temperature)
    text = outputs[0]["generated_text"][-1]["content"]

    rows = [line.strip("-* \t") for line in text.splitlines() if line.strip()]
    dedup = []
    seen = set()
    for row in rows:
        if row not in seen:
            dedup.append(row)
            seen.add(row)
    return dedup[:n_queries]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-passages", type=int, default=10000)
    parser.add_argument("--n-queries", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--generator", choices=["qwen", "template"], default="template")
    parser.add_argument("--qwen-model", type=str, default="Qwen/Qwen3.5-2B")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--output-path", type=str, default="outputs/fineweb_queries.jsonl")
    args = parser.parse_args()

    passages = load_fineweb_sample(n_passages=args.n_passages, seed=args.seed)
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    with open(args.output_path, "w", encoding="utf-8") as f:
        for i, passage_text in enumerate(passages):
            if args.generator == "qwen":
                try:
                    queries = _qwen_queries(
                        passage_text=passage_text,
                        model_id=args.qwen_model,
                        n_queries=args.n_queries,
                        temperature=args.temperature,
                    )
                except Exception as exc:
                    print(f"[warn] qwen failed on passage {i}: {exc}. falling back to template")
                    queries = _template_queries(passage_text, args.n_queries)
            else:
                queries = _template_queries(passage_text, args.n_queries)

            row = {
                "passage_id": f"fineweb_{i}",
                "passage_text": passage_text,
                "queries": queries,
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

            if (i + 1) % 250 == 0:
                print(f"processed {i + 1}/{len(passages)} passages")

    print(f"saved jsonl to {args.output_path}")


if __name__ == "__main__":
    main()

"""FineWeb-Edu loading and sentence tokenization helpers."""

import re

from datasets import load_dataset
import spacy


_NLP = None
_SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?])\s+")


def _get_nlp():
    """Lazily load a spaCy pipeline, falling back to a blank sentencizer."""
    global _NLP
    if _NLP is None:
        try:
            _NLP = spacy.load("en_core_web_sm")
        except OSError:
            nlp = spacy.blank("en")
            nlp.add_pipe("sentencizer")
            _NLP = nlp
    return _NLP


def load_fineweb_sample(n_passages=10000, seed=0, target_tokens=None):
    """Stream FineWeb-Edu and return sampled raw text strings.

    Sampling stops at whichever constraint is hit first:
    - n_passages (if set)
    - target_tokens, using whitespace-token approximation (if set)
    """
    del seed  # Placeholder for deterministic sampling extension.
    ds = load_dataset("HuggingFaceFW/fineweb-edu", split="train", streaming=True)
    texts = []
    token_count = 0
    for item in ds:
        if n_passages is not None and len(texts) >= n_passages:
            break
        if target_tokens is not None and token_count >= target_tokens:
            break
        text = str(item.get("text", "")).strip()
        if text:
            texts.append(text)
            token_count += len(text.split())
    return texts


def split_into_sentences(text, min_len=20):
    """Sentence-tokenize text and filter sentences shorter than min_len."""
    stripped = str(text).strip()
    if not stripped:
        return []

    try:
        nlp = _get_nlp()
        sentences = [s.text.strip() for s in nlp(stripped).sents]
    except Exception:
        sentences = [part.strip() for part in _SENTENCE_BOUNDARY_RE.split(stripped)]

    return [sentence for sentence in sentences if len(sentence) >= min_len]

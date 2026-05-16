"""FineWeb-Edu loading and sentence tokenization helpers."""

from datasets import load_dataset
import spacy


_NLP = None


def _get_nlp():
    """Lazily load spaCy model so import stays light."""
    global _NLP
    if _NLP is None:
        _NLP = spacy.load("en_core_web_sm")
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
    nlp = _get_nlp()
    return [s.text.strip() for s in nlp(text).sents if len(s.text.strip()) >= min_len]

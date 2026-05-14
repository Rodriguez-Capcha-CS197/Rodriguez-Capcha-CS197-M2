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


def load_fineweb_sample(n_passages=10000, seed=0):
    """Stream FineWeb-Edu and return up to n_passages raw text strings."""
    del seed  # Placeholder for deterministic sampling extension.
    ds = load_dataset("HuggingFaceFW/fineweb-edu", split="train", streaming=True)
    texts = []
    for item in ds:
        if len(texts) >= n_passages:
            break
        text = str(item.get("text", "")).strip()
        if text:
            texts.append(text)
    return texts


def split_into_sentences(text, min_len=20):
    """Sentence-tokenize text and filter sentences shorter than min_len."""
    nlp = _get_nlp()
    return [s.text.strip() for s in nlp(text).sents if len(s.text.strip()) >= min_len]

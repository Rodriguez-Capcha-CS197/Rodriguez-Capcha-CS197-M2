"""
Project 10: SKA for Retrieval-Augmented Generation

Goal: Use SKA to encode a retrieval corpus by building a Koopman
operator from document embeddings, then query it with model hidden
states instead of doing vector similarity search. Compare retrieval
quality vs FAISS on a QA dataset.

Requires: single GPU for encoding. FAISS on CPU.
"""

import sys
sys.path.append("..")

import torch
import torch.nn as nn
import numpy as np
from shared.ska import SKAModule, robust_cholesky


class SKARetriever:
    """
    Use SKA's Koopman operator as a retrieval mechanism.

    Instead of storing document embeddings in a FAISS index and doing
    nearest-neighbor search, build a Koopman operator from the document
    embeddings. To retrieve, project the query through the operator.

    The operator compresses the entire corpus into an (r x r) matrix
    per head, instead of storing N x d embeddings.

    TODO: Implement this class.
    """
    def __init__(self, d_model, n_heads=4, rank=32, ridge_eps=1e-3):
        self.d_model = d_model
        self.n_heads = n_heads
        self.rank = rank
        self.ridge_eps = ridge_eps

        self.key_proj = nn.Linear(d_model, n_heads * rank, bias=False)
        self.query_proj = nn.Linear(d_model, n_heads * rank, bias=False)
        self.value_proj = nn.Linear(d_model, n_heads * rank, bias=False)

        nn.init.orthogonal_(self.key_proj.weight)
        nn.init.orthogonal_(self.query_proj.weight)

        self.G = None
        self.C_v = None
        self.L = None
        self.n_docs = 0

    def add_documents(self, embeddings):
        """
        Add document embeddings to the operator.

        Args:
            embeddings: (N, d_model) tensor of document embeddings

        TODO: Implement incremental operator building.
        1. Project embeddings to key space: z = key_proj(embeddings)
        2. Project to value space: v = value_proj(embeddings)
        3. Update G += z^T z (Gram matrix)
        4. Update C_v += v^T z (value readout)
        5. Update Cholesky factor L
        """
        raise NotImplementedError("TODO")

    def retrieve(self, query_embedding, top_k=5):
        """
        Retrieve documents relevant to the query.

        Args:
            query_embedding: (1, d_model) or (d_model,)
            top_k: number of results to return

        Returns:
            scores: (top_k,) relevance scores
            indices: (top_k,) document indices

        TODO: Implement retrieval.
        1. Project query: z_q = query_proj(query_embedding)
        2. Apply operator: output = B_v @ L^{-1} @ z_q
        3. Compare output against stored document value embeddings
        4. Return top_k by similarity

        Note: for the index-based retrieval you'll need to store the
        original document embeddings alongside the operator.
        """
        raise NotImplementedError("TODO")

    def memory_usage(self):
        """Return bytes used by the operator state."""
        r = self.rank
        H = self.n_heads
        if self.G is None:
            return 0
        return (self.G.numel() + self.C_v.numel() + self.L.numel()) * 4


class FAISSRetriever:
    """
    Standard FAISS retrieval baseline.

    TODO: Implement this.
    1. Build a FAISS IndexFlatIP (inner product) index
    2. Add document embeddings
    3. Search with query embedding
    """
    def __init__(self, d_model):
        self.d_model = d_model
        self.index = None
        self.n_docs = 0

    def add_documents(self, embeddings):
        # TODO: add to FAISS index
        raise NotImplementedError("TODO")

    def retrieve(self, query_embedding, top_k=5):
        # TODO: search FAISS index
        raise NotImplementedError("TODO")

    def memory_usage(self):
        """Return bytes used by the index."""
        return self.n_docs * self.d_model * 4


def encode_documents(texts, model_name="sentence-transformers/all-MiniLM-L6-v2"):
    """
    TODO: Encode a list of texts into embeddings using a sentence transformer.
    Return (N, d_model) tensor.
    """
    raise NotImplementedError("TODO")


def load_qa_dataset(n_docs=1000, n_queries=100):
    """
    TODO: Load a QA dataset where each query has known relevant documents.
    Options: Natural Questions, TriviaQA, or SQuAD.
    Return: documents (list of str), queries (list of str),
            relevance (list of list of int) mapping query -> relevant doc indices.
    """
    raise NotImplementedError("TODO")


def run_experiment():
    """
    TODO: Full pipeline.
    1. Load QA dataset with N documents and Q queries
    2. Encode all documents and queries
    3. Build SKARetriever and FAISSRetriever
    4. For each query, retrieve top-5 from both
    5. Compute recall@5 and MRR for both
    6. Compare memory usage: FAISS stores N*d floats, SKA stores r^2*H floats
    7. Sweep rank to find accuracy/memory tradeoff
    8. For N = [100, 500, 1000, 5000, 10000]:
       compare retrieval quality and memory
    """
    print("TODO: implement RAG comparison experiment")


if __name__ == "__main__":
    run_experiment()

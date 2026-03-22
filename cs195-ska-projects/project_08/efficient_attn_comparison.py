"""
Project 8: SKA vs Linear Attention vs Performer vs Hedgehog

Goal: Implement several sub-quadratic attention approximations and
benchmark them all on the same retrieval tasks. Position SKA in the
broader efficient attention landscape.

Requires: single GPU. All methods are easy to implement.
"""

import sys
sys.path.append("..")

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from shared.ska import SKAModule
from shared.utils import evaluate, SmallTransformerLM
from shared.eval_tasks import MQARDataset, PhonebookDataset, InductionDataset


class LinearAttention(nn.Module):
    """
    Linear attention: replace softmax(QK^T) with phi(Q) phi(K)^T.
    Uses ELU+1 as the feature map (Katharopoulos et al., 2020).

    TODO: Implement this.
    1. Project to Q, K, V
    2. Apply feature map: phi(x) = elu(x) + 1
    3. Compute: output = (phi(Q) @ (phi(K)^T @ V)) / (phi(Q) @ phi(K)^T @ 1)
    4. Use causal cumsum for autoregressive version
    """
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.norm = nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, prefix_mask=None):
        # TODO: implement linear attention
        raise NotImplementedError("TODO")


class PerformerAttention(nn.Module):
    """
    Performer: FAVOR+ random feature approximation of softmax attention.
    Uses random orthogonal features (Choromanski et al., 2020).

    TODO: Implement this.
    1. Sample random projection matrix W (d_head, n_features)
    2. phi(x) = exp(x @ W - ||x||^2/2) / sqrt(n_features)
    3. Same linear attention structure as above but with this feature map
    """
    def __init__(self, d_model, n_heads, n_features=64):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.n_features = n_features
        self.norm = nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        W = torch.randn(n_heads, self.head_dim, n_features)
        Q, _ = torch.linalg.qr(W.transpose(-1, -2))
        self.register_buffer("random_features", Q.transpose(-1, -2))

    def forward(self, x, prefix_mask=None):
        # TODO: implement Performer attention
        raise NotImplementedError("TODO")


class CosineReweightedAttention(nn.Module):
    """
    Hedgehog-style: learned feature map that approximates softmax
    attention via a trainable low-rank decomposition.

    TODO: Implement this.
    1. Project Q, K through a learned MLP feature map
    2. Use linear attention with the learned features
    3. The MLP is trained end-to-end
    """
    def __init__(self, d_model, n_heads, feature_dim=64):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.norm = nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.feature_map = nn.Sequential(
            nn.Linear(self.head_dim, feature_dim, bias=False),
            nn.ReLU(),
        )

    def forward(self, x, prefix_mask=None):
        # TODO: implement learned feature map attention
        raise NotImplementedError("TODO")


def build_model_with_method(method_name, d_model=128, n_layers=6,
                            n_heads=4, vocab_size=512):
    """
    TODO: Build a small transformer with the specified attention method
    in layers 2 and 4 (rest use standard causal attention).
    method_name: one of 'attention', 'ska', 'linear', 'performer', 'hedgehog'
    """
    raise NotImplementedError("TODO")


def run_comparison():
    """
    TODO: Main experiment.
    1. Build one model per method, all with same architecture otherwise
    2. Train each on MQAR for 3000 steps
    3. Evaluate all on MQAR, phonebook, induction at context lengths
       [512, 1024, 2048, 4096]
    4. Measure: accuracy, FLOPs per token, state memory
    5. Print comparison table
    6. Plot: accuracy vs context length for each method
    7. Plot: accuracy vs FLOPs Pareto frontier
    """
    methods = ["attention", "ska", "linear", "performer", "hedgehog"]
    tasks = ["mqar", "phonebook", "induction"]
    context_lengths = [512, 1024, 2048, 4096]
    print("TODO: implement comparison")


if __name__ == "__main__":
    run_comparison()

"""
Project 4: SKA Rank vs Retrieval Accuracy Tradeoffs

Goal: Sweep SKA rank from 8 to 128, plot accuracy, memory, FLOPs.
Find the Pareto frontier. Characterize the "knob" SKA provides.

Requires: single GPU or CPU. Runs on small models at short context.
"""

import sys
sys.path.append("..")

import torch
import time
import torch.nn as nn
import torch.nn.functional as F
from shared.ska import SKAModule
from shared.utils import SmallTransformerLM, evaluate, collate_fn, SwiGLUMLP, CausalAttention
from shared.eval_tasks import MQARDataset, PhonebookDataset, InductionDataset
from torch.utils.data import DataLoader


def build_model_with_rank(rank, d_model=128, n_layers=6, n_heads=4, vocab_size=512):
    """
    Build a small transformer where layers 2 and 4 use SKA with the given rank.

    TODO: Implement this.
    Use SmallTransformerLM as a reference, but parameterize the SKA rank.
    """
    raise NotImplementedError("TODO")


def train_model(model, train_dataset, n_steps=2000, lr=3e-4, device="cpu"):
    """
    TODO: Train the model on the given dataset.
    Standard next-token prediction with masked loss.
    Return training loss curve.
    """
    raise NotImplementedError("TODO")


def measure_memory_and_flops(model, seq_len, rank, d_model=128, n_heads=4):
    """
    TODO: Compute theoretical memory and FLOPs for the SKA layers.
    Memory: state size = n_ska_layers * n_heads * (3*rank^2 + P*rank)
    FLOPs per token: n_ska_layers * n_heads * (cholesky_solve + matmuls)
    Compare against attention: state = n_attn_layers * 2 * seq_len * n_heads * head_dim
    Return dict with memory and flops for both SKA and attention.
    """
    raise NotImplementedError("TODO")


def run_sweep():
    """
    TODO: Full rank sweep.
    1. For rank in [8, 16, 24, 32, 48, 64, 96, 128]:
       a. Build model with that rank
       b. Train on MQAR for 2000 steps
       c. Evaluate accuracy on MQAR, phonebook, induction
       d. Measure memory and FLOPs
    2. Plot: accuracy vs rank, memory vs rank, FLOPs vs rank
    3. Plot: Pareto frontier (accuracy vs memory)
    4. Compare against attention baseline at each context length
    """
    ranks = [8, 16, 24, 32, 48, 64, 96, 128]
    context_lengths = [512, 1024, 2048, 4096]
    print("TODO: implement sweep")


if __name__ == "__main__":
    run_sweep()

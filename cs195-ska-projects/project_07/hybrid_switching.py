"""
Project 7: Hybrid Context Switching (Attention <-> SKA)

Goal: Implement dynamic switching: use normal attention at short
context, switch to SKA when context crosses a threshold. Measure
the crossover point where SKA becomes faster/cheaper than
FlashAttention on actual hardware.

Requires: single GPU with CUDA for timing (Colab T4 or better).
"""

import sys
sys.path.append("..")

import torch
import torch.nn as nn
import time
from shared.ska import SKAModule
from shared.utils import CausalAttention


class HybridAttentionLayer(nn.Module):
    """
    Layer that dynamically switches between standard attention and
    SKA based on the current sequence length.

    TODO: Implement this.
    Contains both a CausalAttention module and an SKAModule.
    In forward(), check the sequence length:
      - If T <= threshold: use CausalAttention
      - If T > threshold: use SKA

    The threshold is a configurable parameter.
    """
    def __init__(self, d_model, n_heads, ska_rank=32, threshold=4096):
        super().__init__()
        self.threshold = threshold
        self.attention = CausalAttention(d_model, n_heads)
        self.ska = SKAModule(d_model, n_heads, rank=ska_rank)

    def forward(self, x, prefix_mask=None):
        # TODO: implement dynamic switching
        raise NotImplementedError("TODO")


class HybridAttentionLayerSmooth(nn.Module):
    """
    Smooth blending variant: instead of a hard switch, blend the
    outputs of attention and SKA with a learned or scheduled weight.

    TODO: Implement this.
    alpha = sigmoid(learned_param) or alpha = f(seq_len)
    output = alpha * attention_output + (1 - alpha) * ska_output
    """
    def __init__(self, d_model, n_heads, ska_rank=32):
        super().__init__()
        self.attention = CausalAttention(d_model, n_heads)
        self.ska = SKAModule(d_model, n_heads, rank=ska_rank)
        self.blend_param = nn.Parameter(torch.tensor(0.0))

    def forward(self, x, prefix_mask=None):
        # TODO: implement smooth blending
        raise NotImplementedError("TODO")


def benchmark_latency(d_model=256, n_heads=8, ska_rank=32, batch_size=1,
                      device="cuda"):
    """
    TODO: Measure forward pass latency for attention vs SKA across
    context lengths.

    1. For seq_len in [512, 1024, 2048, 4096, 8192, 16384, 32768]:
       a. Create random input (batch_size, seq_len, d_model)
       b. Time attention forward (use torch.cuda.Event for precise timing)
       c. Time SKA forward
       d. Record both times
    2. Find the crossover point where SKA becomes faster
    3. Return results dict
    """
    raise NotImplementedError("TODO")


def benchmark_memory(d_model=256, n_heads=8, ska_rank=32, device="cuda"):
    """
    TODO: Measure peak GPU memory for attention vs SKA.

    1. For each seq_len, run forward + backward
    2. Record torch.cuda.max_memory_allocated()
    3. Attention memory should scale linearly with seq_len
    4. SKA memory should stay roughly constant
    5. Find crossover point
    """
    raise NotImplementedError("TODO")


def train_hybrid_model(threshold=4096, n_steps=2000):
    """
    TODO: Train a small model with HybridAttentionLayer on MQAR.
    Vary the training context length to span both sides of the
    threshold. Show that the model works seamlessly across the switch.
    """
    raise NotImplementedError("TODO")


def run_experiment():
    """
    TODO: Full experiment.
    1. Run latency benchmark, find crossover point
    2. Run memory benchmark, find crossover point
    3. Train hybrid model, evaluate at multiple context lengths
    4. Plot: latency vs seq_len (attention, SKA, hybrid)
    5. Plot: memory vs seq_len (attention, SKA, hybrid)
    6. Table: accuracy at each context length for all three
    """
    print("TODO: implement hybrid switching experiment")


if __name__ == "__main__":
    run_experiment()

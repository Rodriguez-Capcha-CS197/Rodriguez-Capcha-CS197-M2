"""
Project 3: Koopman MLP Ablation on GPT-2

Goal: Replace MLP layers in GPT-2 Small (124M) with Koopman MLP,
measure perplexity without training, then fine-tune briefly and
measure recovery.

Requires: ~8GB GPU (Colab free tier T4 works).
"""

import sys
sys.path.append("..")

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from shared.koopman_mlp import SpectralKoopmanMLP, SpectralKoopmanMLPGated


def load_gpt2():
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def measure_perplexity(model, tokenizer, max_samples=200):
    """
    TODO: Measure perplexity on wikitext-2 validation set.
    1. Load wikitext-2-raw-v1 validation split
    2. Tokenize into 1024-token chunks
    3. Forward pass, compute cross-entropy loss
    4. Return exp(avg_loss)
    """
    raise NotImplementedError("TODO")


def replace_mlp_layers(model, layer_indices=None, gated=False):
    """
    TODO: Replace GPT-2 MLP layers with Koopman MLP.
    GPT-2 MLP: model.transformer.h[i].mlp has c_fc (768->3072), c_proj (3072->768)
    1. Create SpectralKoopmanMLP(d=768) or gated variant
    2. Initialize lift from c_fc weights, readout from c_proj
    3. Initialize gamma=1, omega=0 (identity = no change at start)
    4. Replace the module in place
    """
    raise NotImplementedError("TODO")


def finetune(model, tokenizer, n_steps=500, lr=1e-4):
    """
    TODO: Fine-tune only the Koopman MLP parameters.
    1. Freeze everything except gamma, omega, lift, readout
    2. Train for n_steps on wikitext-2 train
    3. Return model
    """
    raise NotImplementedError("TODO")


def run_ablation():
    """
    TODO: Full pipeline.
    1. Baseline GPT-2 perplexity
    2. Replace all MLPs -> Koopman MLP, measure (no training)
    3. Replace all MLPs -> Gated variant, measure (no training)
    4. Fine-tune each for 500 steps, measure again
    5. Print parameter counts and perplexity table
    """
    print("TODO: implement ablation pipeline")


if __name__ == "__main__":
    run_ablation()

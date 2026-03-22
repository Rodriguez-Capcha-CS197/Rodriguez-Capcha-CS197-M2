"""
Project 2: SKA as Drop-In for Pretrained Attention

Goal: Take a small pretrained language model (Qwen2.5-0.5B or
Llama-3.2-1B), extract Q/K/V/O weight matrices from its attention
layers, initialize SKA projections from them, and measure
needle-in-a-haystack retrieval accuracy vs the original.

No training needed. Just weight transplant and eval.

Requires: ~4GB GPU (Colab free tier works for 0.5B model).
"""

import sys
sys.path.append("..")

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from shared.ska import SKAModule


def extract_attention_weights(model):
    """
    Walk a HuggingFace model and extract Q/K/V/O weights from each
    attention layer.

    TODO: Implement this function.
    1. Find all attention layers (look for 'self_attn' attribute)
    2. For each, extract q_proj, k_proj, v_proj, o_proj weights
    3. Handle GQA: note which layers have fewer KV heads than Q heads
    4. Return a list of dicts, one per attention layer

    Hint: for Qwen2.5-0.5B the path is model.model.layers[i].self_attn
    """
    layers_info = []
    # TODO: fill in
    raise NotImplementedError("TODO: extract attention weights")
    return layers_info


def init_ska_from_attention(attn_weights, ska_rank=32):
    """
    Initialize an SKA module's projections from pretrained attention weights.

    Args:
        attn_weights: dict with 'q_proj', 'k_proj', 'v_proj', 'o_proj'
                      and metadata like 'n_heads', 'n_kv_heads', 'head_dim'
        ska_rank: rank for the SKA module

    Returns:
        SKAModule with initialized weights

    TODO: Implement the weight mapping.

    For the key_proj: SKA's key space is rank-r, not head_dim.
    Option A: Use SVD on the original K projection to find the top-r
              directions, initialize SKA key_proj from those.
    Option B: Use a random projection from head_dim to r, initialize
              key_proj = random_proj @ original_k_proj.
    Option C: If rank >= head_dim, initialize directly and zero-pad.

    For GQA: the original model has fewer KV heads than Q heads.
    Decide how to map this to SKA's head structure. See the README
    for three options.
    """
    raise NotImplementedError("TODO: initialize SKA from attention weights")


def replace_attention_layer(model, layer_idx, ska_module):
    """
    Replace a single attention layer in the model with an SKA module.

    TODO: Implement this.
    1. Get the original layer: model.model.layers[layer_idx]
    2. Create a wrapper that uses SKA instead of self_attn
    3. Preserve the original layernorms and MLP
    4. Replace the layer in the model
    """
    raise NotImplementedError("TODO: replace attention layer")


def needle_in_haystack_test(model, tokenizer, context_length=4096,
                             n_trials=20):
    """
    Place a specific fact ("The secret number is 42") at a random
    position in a long context of filler text, then ask the model
    to recall it.

    TODO: Implement this.
    1. Generate filler text (repeated paragraphs, random text, etc.)
    2. Insert the needle at a random position
    3. Append the query at the end: "What is the secret number?"
    4. Generate and check if the model outputs "42"
    5. Repeat n_trials times at different positions
    6. Return accuracy (fraction of correct retrievals)

    Note: for a 0.5B model, accuracy won't be perfect even with
    normal attention. The point is to compare SKA vs attention,
    not to achieve 100%.
    """
    raise NotImplementedError("TODO: implement needle in haystack")


def run_comparison(model_name="Qwen/Qwen2.5-0.5B-Instruct"):
    """
    Full pipeline: load model, transplant weights, compare.

    TODO: Complete this function.
    1. Load the pretrained model and tokenizer
    2. Run needle_in_haystack_test with original attention
    3. Extract attention weights
    4. For each attention layer, create SKA and transplant weights
    5. Replace attention layers with SKA
    6. Run needle_in_haystack_test again with SKA
    7. Print comparison table:
        - Accuracy at each context length
        - Peak memory usage
        - Time per forward pass
    """
    print("TODO: implement full comparison pipeline")


if __name__ == "__main__":
    run_comparison()

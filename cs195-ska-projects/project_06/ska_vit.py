"""
Project 6: SKA for Vision Transformers

Goal: Replace attention in a small ViT (ViT-Tiny or ViT-Small) with
SKA. Images have a natural prefix (patch embeddings) and query (CLS
token). Evaluate on ImageNet classification or CIFAR-100.

No long-context issues, everything fits on Colab.

Requires: single GPU (Colab T4 works).
"""

import sys
sys.path.append("..")

import torch
import torch.nn as nn
from shared.ska import SKAModule


def load_vit(model_name="google/vit-base-patch16-224"):
    """
    TODO: Load a pretrained ViT from HuggingFace.
    Start with vit-base-patch16-224 or the smaller
    WinKawaks/vit-tiny-patch16-224 if memory is tight.
    Return model and feature_extractor/processor.
    """
    from transformers import ViTForImageClassification, ViTImageProcessor
    raise NotImplementedError("TODO: load model and processor")


def extract_vit_attention_weights(model):
    """
    TODO: Extract Q/K/V/O weights from each ViT attention layer.
    ViT attention is at model.vit.encoder.layer[i].attention.attention
    with query, key, value projections and output.dense.
    Return list of weight dicts.
    """
    raise NotImplementedError("TODO")


def build_ska_vit_layer(original_layer, d_model, n_heads, ska_rank=32):
    """
    TODO: Create a replacement ViT encoder layer that uses SKA.
    1. Keep the original LayerNorms and MLP
    2. Replace the self-attention with SKA
    3. Initialize SKA projections from original Q/K/V weights
    4. Return the new layer

    Key difference from language: ViT attention is NOT causal.
    All patches attend to all patches. For SKA this means
    prefix_mask should be all-ones (every token is "context").
    """
    raise NotImplementedError("TODO")


def replace_vit_attention(model, ska_rank=32):
    """
    TODO: Replace all attention layers in the ViT with SKA.
    Walk model.vit.encoder.layer, replace each attention block.
    Return modified model.
    """
    raise NotImplementedError("TODO")


def evaluate_imagenet(model, processor, n_samples=1000):
    """
    TODO: Evaluate on ImageNet validation set (or CIFAR-100 if
    ImageNet isn't available).
    1. Load dataset (use datasets library or torchvision)
    2. Run inference, compute top-1 and top-5 accuracy
    3. Return accuracy dict
    """
    raise NotImplementedError("TODO")


def run_experiment():
    """
    TODO: Full pipeline.
    1. Load pretrained ViT, measure baseline accuracy
    2. Replace attention with SKA at various ranks
    3. Measure accuracy (no fine-tuning) for each rank
    4. Fine-tune SKA params only for 1 epoch on training set
    5. Measure accuracy after fine-tuning
    6. Compare: accuracy vs rank, accuracy vs parameter count
    """
    print("TODO: implement ViT experiment")


if __name__ == "__main__":
    run_experiment()

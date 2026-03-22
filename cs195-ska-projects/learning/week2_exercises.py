"""
Week 2 Coding Exercises

Run this file to check your work on the hands-on exercises.
Fill in the TODO sections, then run: python week2_exercises.py
"""

import sys
sys.path.append("..")

import torch
import torch.nn.functional as F


def exercise_1_manual_ska():
    """Build SKA from scratch without importing the module."""
    print("Exercise 1: Manual SKA")
    torch.manual_seed(42)

    d_model, rank, P = 8, 4, 6
    seq_len = 20

    W_key = torch.randn(rank, d_model)
    W_query = torch.randn(rank, d_model)
    W_value = torch.randn(P, d_model)

    x = torch.randn(seq_len, d_model)

    # TODO: implement steps c through k from the exercise
    # z = ...
    # v = ...
    # G = ...
    # M = ...
    # C_v = ...
    # G_tilde = ...
    # L = ...
    # A_w = ...
    # B_v = ...
    # For a query (last token): output = B_v @ A_w @ L^{-1} @ z_q
    raise NotImplementedError("TODO: implement manual SKA")


def exercise_2_compare_to_module():
    """Compare manual SKA output to SKAModule."""
    print("Exercise 2: Compare to SKAModule")
    from shared.ska import SKAModule

    torch.manual_seed(42)
    d_model, n_heads, rank = 8, 1, 4

    ska = SKAModule(d_model, n_heads, rank=rank, head_dim=6)
    x = torch.randn(1, 20, d_model)

    # TODO: run SKA forward, get output
    # TODO: compare to your Exercise 1 result
    raise NotImplementedError("TODO: compare outputs")


def exercise_3_gram_matrix_visualization():
    """Visualize Gram matrices for different input types."""
    print("Exercise 3: Gram matrix visualization")
    torch.manual_seed(42)

    d, r = 8, 4
    W_key = torch.randn(r, d)

    # TODO: generate three types of sequences
    # a. Random noise
    # b. Repeating pattern
    # c. All-same token
    # For each, compute z = x @ W_key.T, then G = z.T @ z
    # Print or plot each G
    raise NotImplementedError("TODO: compute and visualize Gram matrices")


def exercise_5_rotation_blocks():
    """Understand the 2x2 rotation in Koopman MLP."""
    print("Exercise 5: Rotation blocks")

    def rotate(g1, g2, gamma, omega):
        z1 = gamma * g1 + omega * g2
        z2 = -omega * g1 + gamma * g2
        return z1, z2

    # TODO: test with the four cases from the exercise
    # a. gamma=1, omega=0 (identity)
    # b. gamma=0, omega=1 (90 degrees)
    # c. gamma=0.7, omega=0.7 (45 degrees, magnitude=~0.99)
    # d. Apply spectral normalization when radius > 1

    g1, g2 = 1.0, 0.0

    # TODO: apply each rotation, print results
    raise NotImplementedError("TODO: test rotations")


def exercise_6_parameter_count():
    """Compare Koopman MLP vs SwiGLU parameter counts."""
    print("Exercise 6: Parameter counts")
    from shared.koopman_mlp import SpectralKoopmanMLP
    from shared.utils import SwiGLUMLP

    d = 64
    koopman = SpectralKoopmanMLP(d)
    swiglu = SwiGLUMLP(d)

    k_params = sum(p.numel() for p in koopman.parameters())
    s_params = sum(p.numel() for p in swiglu.parameters())

    print(f"  Koopman MLP: {k_params:,} params")
    print(f"  SwiGLU MLP:  {s_params:,} params")
    print(f"  Ratio: {k_params/s_params:.2%}")
    print(f"  Savings: {1 - k_params/s_params:.2%}")


def exercise_8_build_models():
    """Build three model variants and compare."""
    print("Exercise 8: Build models")
    from shared.utils import SmallTransformerLM

    configs = [
        ("All attention", []),
        ("SKA layers 1,3", [1, 3]),
        ("SKA + Koopman MLP", [1, 3]),
    ]

    for name, ska_indices in configs:
        use_koopman = (name == "SKA + Koopman MLP")
        model = SmallTransformerLM(
            vocab_size=512, d_model=64, n_layers=4, n_heads=4,
            ska_layer_indices=ska_indices, use_koopman_mlp=use_koopman,
        )
        n_params = sum(p.numel() for p in model.parameters())
        x = torch.randint(0, 512, (2, 32))
        out = model(x)
        print(f"  {name}: {n_params:,} params, logits shape {out['logits'].shape}")


def exercise_9_train_mqar():
    """Train and evaluate on MQAR."""
    print("Exercise 9: Train on MQAR")
    from shared.eval_tasks import MQARDataset
    from shared.utils import SmallTransformerLM, evaluate, collate_fn
    from torch.utils.data import DataLoader

    torch.manual_seed(42)

    train_ds = MQARDataset(n_examples=500, M=4, seq_len=128, vocab_size=512)
    eval_ds = MQARDataset(n_examples=100, M=4, seq_len=128, vocab_size=512)

    # TODO: build model, train for 1000 steps, evaluate
    # Try all three variants from Exercise 8
    # Print accuracy for each
    raise NotImplementedError("TODO: train and evaluate")


def exercise_10_prefix_mask():
    """Test the effect of correct prefix masking."""
    print("Exercise 10: Prefix mask experiment")

    # TODO: train SKA model with correct prefix_mask vs all-ones
    # Compare accuracy
    raise NotImplementedError("TODO: test prefix mask effect")


if __name__ == "__main__":
    exercises = [
        exercise_1_manual_ska,
        exercise_2_compare_to_module,
        exercise_3_gram_matrix_visualization,
        exercise_5_rotation_blocks,
        exercise_6_parameter_count,
        exercise_8_build_models,
        exercise_9_train_mqar,
        exercise_10_prefix_mask,
    ]

    for ex in exercises:
        print()
        try:
            ex()
        except NotImplementedError as e:
            print(f"  {e}")
        print()

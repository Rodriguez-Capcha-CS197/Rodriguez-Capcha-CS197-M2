import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


def collate_fn(batch):
    max_len = max(ex["input_ids"].shape[0] for ex in batch)
    B = len(batch)
    out = {
        "input_ids": torch.zeros(B, max_len, dtype=torch.long),
        "labels": torch.zeros(B, max_len, dtype=torch.long),
        "loss_mask": torch.zeros(B, max_len),
        "prefix_mask": torch.zeros(B, max_len),
    }
    for i, ex in enumerate(batch):
        T = ex["input_ids"].shape[0]
        for k in out:
            out[k][i, :T] = ex[k]
    return out


def evaluate(model, dataset, device="cpu", batch_size=8):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        collate_fn=collate_fn, drop_last=False)
    model.eval()
    correct, total, total_loss = 0, 0, 0.0
    with torch.no_grad():
        for batch in loader:
            ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            lm = batch["loss_mask"].to(device)
            pm = batch["prefix_mask"].to(device)
            logits = model(ids, prefix_mask=pm)
            if isinstance(logits, dict):
                logits = logits["logits"]
            preds = logits.argmax(dim=-1)
            correct += ((preds == labels) * lm).sum().item()
            total += lm.sum().item()
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1), reduction="none"
            ).view_as(labels)
            total_loss += (loss * lm).sum().item()
    acc = correct / max(total, 1)
    avg_loss = total_loss / max(total, 1)
    return {"accuracy": acc, "loss": avg_loss}


class SwiGLUMLP(nn.Module):
    """Standard SwiGLU for baseline comparisons."""
    def __init__(self, d, expand=2.667):
        super().__init__()
        d_ff = ((int(d * expand) + 63) // 64) * 64
        self.norm = nn.LayerNorm(d)
        self.w1 = nn.Linear(d, d_ff, bias=False)
        self.w2 = nn.Linear(d, d_ff, bias=False)
        self.w3 = nn.Linear(d_ff, d, bias=False)

    def forward(self, x):
        h = self.norm(x)
        return x + self.w3(F.silu(self.w1(h)) * self.w2(h))


class CausalAttention(nn.Module):
    """Standard causal multi-head attention for baselines."""
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.norm = nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, prefix_mask=None):
        B, T, d = x.shape
        H, D = self.n_heads, self.head_dim
        h = self.norm(x)
        qkv = self.qkv(h).reshape(B, T, 3, H, D)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
        scores = torch.einsum('bthd,bshd->bhts', q, k) / math.sqrt(D)
        causal = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(causal.unsqueeze(0).unsqueeze(0), float('-inf'))
        attn = F.softmax(scores, dim=-1)
        out = torch.einsum('bhts,bshd->bthd', attn, v).reshape(B, T, d)
        return x + self.proj(out)


class SmallTransformerLM(nn.Module):
    """
    Tiny transformer for testing on Colab. No Mamba dependency.
    Can swap attention for SKA by passing use_ska=True.
    """
    def __init__(self, vocab_size=512, d_model=128, n_layers=6,
                 n_heads=4, ska_layer_indices=None, ska_rank=24,
                 use_koopman_mlp=False):
        super().__init__()
        from shared.ska import SKAModule
        from shared.koopman_mlp import SpectralKoopmanMLP

        self.embed = nn.Embedding(vocab_size, d_model)
        ska_set = set(ska_layer_indices or [])

        self.seq_layers = nn.ModuleList()
        self.mlp_layers = nn.ModuleList()
        for i in range(n_layers):
            if i in ska_set:
                self.seq_layers.append(SKAModule(d_model, n_heads, rank=ska_rank))
            else:
                self.seq_layers.append(CausalAttention(d_model, n_heads))
            if use_koopman_mlp:
                self.mlp_layers.append(SpectralKoopmanMLP(d_model))
            else:
                self.mlp_layers.append(SwiGLUMLP(d_model))

        self.norm_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.embed.weight

    def forward(self, input_ids, labels=None, prefix_mask=None):
        h = self.embed(input_ids)
        for seq, mlp in zip(self.seq_layers, self.mlp_layers):
            if hasattr(seq, 'forward'):
                import inspect
                sig = inspect.signature(seq.forward)
                if 'prefix_mask' in sig.parameters:
                    h = seq(h, prefix_mask=prefix_mask)
                else:
                    h = seq(h)
            h = mlp(h)
        logits = self.lm_head(self.norm_f(h))
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits[:, :-1].contiguous().view(-1, logits.size(-1)),
                labels[:, 1:].contiguous().view(-1), ignore_index=-100)
        return {"loss": loss, "logits": logits}

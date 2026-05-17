"""Save/load utilities for sweep records, checkpoints, and held-out test sets."""

import json
import os
import torch


def save_records(records, path):
    output_dir = os.path.dirname(path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)
    print(f"Saved {len(records)} records to: {path}")


def load_records(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_held_out_qids(qids, path):
    output_dir = os.path.dirname(path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(list(qids), f, indent=2)
    print(f"Saved {len(qids)} held-out qids to: {path}")


def load_held_out_qids(path):
    with open(path, "r", encoding="utf-8") as f:
        return set(json.load(f))


def save_classifier_checkpoint(model, metadata, path):
    output_dir = os.path.dirname(path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(), **metadata}, path)
    print(f"Saved checkpoint to: {path}")


def load_classifier_checkpoint(path, model_cls):
    ckpt = torch.load(path, weights_only=False)
    model = model_cls(input_dim=ckpt["input_dim"], hidden_dim=ckpt["hidden_dim"])
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt

"""LambdaPredictor MLP. Classifier head over the lambda grid."""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from .constants import LAMBDA_GRID, MODES
from .scoring import ensure_1d


LAMBDA_TO_IDX = {lam: i for i, lam in enumerate(LAMBDA_GRID)}
IDX_TO_LAMBDA = {i: lam for i, lam in enumerate(LAMBDA_GRID)}


class LambdaPredictor(nn.Module):
    """Classifier MLP over the lambda grid."""

    def __init__(self, input_dim, hidden_dim=64, num_classes=len(LAMBDA_GRID)):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        return self.net(x)  # logits over lambda grid

    def predict_lambda(self, x):
        """Return the predicted lambda value (snapped to grid)."""
        with torch.no_grad():
            logits = self.forward(x)
            idx = logits.argmax(dim=-1).cpu().numpy()
        return np.array([IDX_TO_LAMBDA[int(i)] for i in idx])

    def predict_proba(self, x):
        """Return softmax probabilities over the grid."""
        with torch.no_grad():
            return torch.softmax(self.forward(x), dim=-1).cpu().numpy()


def build_training_dataset(records, embed_fn):
    from shared.scoring import ensure_1d
    optimal = [r for r in records if r["is_optimal"]]
    queries = [r["query"] for r in optimal]
    modes = [r["mode"] for r in optimal]
    y = np.asarray([r["lambda"] for r in optimal], dtype=np.float32)
    embeddings = [ensure_1d(embed_fn(q)) for q in queries]
    X = np.asarray(embeddings, dtype=np.float32)
    return X, y, queries, modes


def stratified_train_val_split(X, y, modes, val_frac=0.2, seed=0):
    rng = np.random.default_rng(seed)
    modes_arr = np.asarray(modes)
    train_idx, val_idx = [], []

    for mode in sorted(set(modes)):
        mode_indices = np.where(modes_arr == mode)[0]
        rng.shuffle(mode_indices)
        n_val = max(1, int(len(mode_indices) * val_frac))
        val_idx.extend(mode_indices[:n_val])
        train_idx.extend(mode_indices[n_val:])

    train_idx = np.asarray(train_idx)
    val_idx = np.asarray(val_idx)
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    return X[train_idx], X[val_idx], y[train_idx], y[val_idx], train_idx, val_idx


def train_classifier(X_train, y_train, X_val, y_val, hidden_dim=64, num_epochs=200, lr=1e-3, seed=0):
    """Train a LambdaPredictor classifier. Returns (model, train_losses, val_losses)."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    input_dim = X_train.shape[1]
    model = LambdaPredictor(input_dim=input_dim, hidden_dim=hidden_dim)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.long)

    train_losses, val_losses = [], []
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(X_train_t)
        train_loss = criterion(logits, y_train_t)
        train_loss.backward()
        optimizer.step()
        train_losses.append(float(train_loss.item()))

        model.eval()
        with torch.no_grad():
            val_logits = model(X_val_t)
            val_loss = criterion(val_logits, y_val_t)
        val_losses.append(float(val_loss.item()))

        if epoch % 50 == 0 or epoch == num_epochs - 1:
            train_acc = (logits.argmax(-1) == y_train_t).float().mean().item()
            val_acc = (val_logits.argmax(-1) == y_val_t).float().mean().item()
            print(f"Epoch {epoch:03d} | train CE={train_loss.item():.4f} acc={train_acc:.3f} | "
                  f"val CE={val_loss.item():.4f} acc={val_acc:.3f}")

    return model, train_losses, val_losses


def evaluate_classifier(model, X, y_true_idx):
    """Top-1 and top-1-within-1-grid-step accuracy."""
    X_t = torch.tensor(X, dtype=torch.float32)
    with torch.no_grad():
        pred_idx = model(X_t).argmax(-1).numpy()
    top1 = float(np.mean(pred_idx == y_true_idx))
    within1 = float(np.mean(np.abs(pred_idx - y_true_idx) <= 1))
    return {"top1_accuracy": top1, "within_one_grid_step": within1, "pred_idx": pred_idx}
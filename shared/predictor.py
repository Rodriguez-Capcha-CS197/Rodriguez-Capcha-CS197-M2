"""LambdaPredictor MLP. Regressor head predicting log(lambda)."""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from .constants import EPS
from .scoring import ensure_1d


class LambdaPredictor(nn.Module):
    """Regressor MLP that predicts log(lambda). Supports arbitrary input dimensions."""

    def __init__(self, input_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)

    def predict_lambda(self, x):
        """
        Inference helper: Pass a tensor, get the actual positive lambda float(s).
        (Exponentiates the log-space output back to linear space).
        """
        self.eval()
        with torch.no_grad():
            log_lambda = self.forward(x)
            actual_lambda = torch.exp(log_lambda)
        return actual_lambda.cpu().numpy()


def build_training_dataset(records, embed_fn):
    """Convert optimal records into regression training data."""
    optimal = [r for r in records if r["is_optimal"]]
    queries = [r["query"] for r in optimal]
    modes = [r["mode"] for r in optimal]
    y = np.asarray([r["lambda"] for r in optimal], dtype=np.float32)
    embeddings = [ensure_1d(embed_fn(q)) for q in queries]
    X = np.asarray(embeddings, dtype=np.float32)
    return X, y, queries, modes


def stratified_train_val_split(X, y, modes, val_frac=0.2, seed=0):
    """Stratify by query mode so validation has examples from each mode."""
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


def train_predictor(X_train, y_train, X_val, y_val, hidden_dim=64, num_epochs=200, lr=1e-3, seed=0):
    """Train a LambdaPredictor regressor in log-space. Returns (model, train_losses, val_losses)."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    input_dim = X_train.shape[1]
    model = LambdaPredictor(input_dim=input_dim, hidden_dim=hidden_dim)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    # Convert targets to log-space for training automatically
    y_train_log_t = torch.tensor(np.log(np.clip(y_train, EPS, None)), dtype=torch.float32)

    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    y_val_log_t = torch.tensor(np.log(np.clip(y_val, EPS, None)), dtype=torch.float32)

    train_losses, val_losses = [], []
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        preds_log = model(X_train_t)
        train_loss = criterion(preds_log, y_train_log_t)
        train_loss.backward()
        optimizer.step()
        train_losses.append(float(train_loss.item()))

        model.eval()
        with torch.no_grad():
            val_preds_log = model(X_val_t)
            val_loss = criterion(val_preds_log, y_val_log_t)
        val_losses.append(float(val_loss.item()))

        if epoch % 50 == 0 or epoch == num_epochs - 1:
            print(f"Epoch {epoch:03d} | train log-MSE={train_loss.item():.4f} | val log-MSE={val_loss.item():.4f}")

    return model, train_losses, val_losses


def evaluate_predictor(model, X, y_true):
    """Evaluate the regressor on MSE and MAE in linear space."""
    X_t = torch.tensor(X, dtype=torch.float32)
    pred_lambda = model.predict_lambda(X_t)

    mse = float(np.mean((pred_lambda - y_true) ** 2))
    mae = float(np.mean(np.abs(pred_lambda - y_true)))

    return {"mse": mse, "mae": mae, "pred_lambda": pred_lambda}
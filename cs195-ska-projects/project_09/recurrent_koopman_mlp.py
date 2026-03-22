"""
Project 9: Recurrent Koopman MLP

Goal: Make the Koopman MLP recurrent by letting gamma/omega evolve
over the sequence. Each token updates the eigenvalues based on the
hidden state, creating a nonlinear recurrent transform with O(d_k)
state. Compare against TTT-MLP on retrieval tasks.

Requires: single GPU for training small models.
"""

import sys
sys.path.append("..")

import torch
import torch.nn as nn
import torch.nn.functional as F
from shared.koopman_mlp import SpectralKoopmanMLP
from shared.utils import evaluate
from shared.eval_tasks import MQARDataset, SystemPromptDataset


class RecurrentKoopmanMLP(nn.Module):
    """
    Recurrent Koopman MLP: gamma and omega are updated at each token
    based on the hidden state, making the rotation adapt to context.

    The state is (gamma, omega) with shape (d_k//2,) each = O(d_k) total.
    Compare to TTT-MLP where the state is the full MLP weight matrix = O(d_k^2).

    TODO: Implement this.
    Architecture:
      1. Lift: g = SiLU(W_lift @ x)
      2. Update eigenvalues from hidden state:
         delta_gamma, delta_omega = small_network(x)
         gamma_t = gamma_{t-1} + lr * delta_gamma
         omega_t = omega_{t-1} + lr * delta_omega
      3. Rotate with current eigenvalues: z = rotate(g, gamma_t, omega_t)
      4. Readout: out = W_readout @ z
    """
    def __init__(self, d, expand=2.667, spectral_norm_gamma=True):
        super().__init__()
        self.d_k = ((int(d * expand) + 63) // 64) * 64
        self.spectral_norm_gamma = spectral_norm_gamma

        self.norm = nn.LayerNorm(d)
        self.lift = nn.Linear(d, self.d_k, bias=False)
        self.readout = nn.Linear(self.d_k, d, bias=False)

        self.gamma_init = nn.Parameter(torch.ones(self.d_k // 2))
        self.omega_init = nn.Parameter(torch.zeros(self.d_k // 2))

        # TODO: define the eigenvalue update network
        # This should map from d_model to (d_k//2 * 2) to produce
        # delta_gamma and delta_omega at each token.
        # Keep it small: one linear layer or a tiny MLP.
        self.eigenvalue_updater = None  # TODO

        self.update_lr = nn.Parameter(torch.tensor(0.01))

    def forward(self, x):
        """
        TODO: Implement the recurrent forward pass.

        For training efficiency, process all tokens in parallel but
        use a cumulative sum to simulate the sequential eigenvalue updates:

        Option A (simple, sequential):
          gamma, omega = self.gamma_init, self.omega_init
          outputs = []
          for t in range(T):
              delta = self.eigenvalue_updater(x[:, t])
              gamma = gamma + self.update_lr * delta[:, :d_k//2]
              omega = omega + self.update_lr * delta[:, d_k//2:]
              g = SiLU(self.lift(self.norm(x[:, t])))
              z = rotate(g, gamma, omega)
              outputs.append(self.readout(z))
          return x + stack(outputs)

        Option B (parallel approximation):
          Compute all deltas at once, cumsum to get gamma_t/omega_t,
          then apply rotation in parallel. This is an approximation
          because the updater sees x not the recurrent state, but
          it's much faster for training.
        """
        raise NotImplementedError("TODO")

    def _rotate(self, g, gamma, omega):
        """Apply the 2x2 block rotation."""
        pair = g.view(*g.shape[:-1], self.d_k // 2, 2)
        g1, g2 = pair[..., 0], pair[..., 1]
        if self.spectral_norm_gamma:
            radius = torch.sqrt(gamma * gamma + omega * omega).clamp(min=1e-8)
            scale = torch.clamp(radius, max=1.0) / radius
            gamma = gamma * scale
            omega = omega * scale
        z1 = gamma * g1 + omega * g2
        z2 = -omega * g1 + gamma * g2
        return torch.stack([z1, z2], dim=-1).reshape_as(g)


class SimpleTTTMLP(nn.Module):
    """
    Simplified TTT-MLP baseline for comparison.
    Hidden state is a small MLP that gets updated by gradient descent
    at each token.

    TODO: Implement this.
    1. Inner model: Linear(d, d) or small 2-layer MLP
    2. At each token: compute self-supervised loss, take gradient step
    3. Use the updated model to transform the token
    """
    def __init__(self, d, inner_lr=0.01):
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.inner_model = nn.Linear(d, d, bias=False)
        self.inner_lr = inner_lr

    def forward(self, x):
        # TODO: implement TTT-MLP forward with gradient steps
        raise NotImplementedError("TODO")


def compare_recurrent_mlps():
    """
    TODO: Main experiment.
    1. Build small models (6 layers, d=128) with:
       a. Standard SwiGLU MLP (baseline)
       b. Static Koopman MLP (no recurrence)
       c. Recurrent Koopman MLP (this project)
       d. Simple TTT-MLP
    2. Train all on MQAR and SystemPrompt tasks
    3. Evaluate at context lengths [512, 1024, 2048, 4096]
    4. Compare: accuracy, state size, FLOPs per token
    5. Key question: does the recurrence in Koopman MLP help retrieval?
       Static Koopman MLP is feedforward (no context accumulation).
       Recurrent version should accumulate context like TTT-MLP but
       with O(d_k) state instead of O(d_k^2).
    """
    print("TODO: implement recurrent MLP comparison")


if __name__ == "__main__":
    compare_recurrent_mlps()

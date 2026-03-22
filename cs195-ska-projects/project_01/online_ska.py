"""
Project 1: Online SKA with Rank-1 Updates

Goal: Implement an incremental version of SKA where the sufficient
statistics (G, M, C_v) are running sums updated one token at a time,
and the Cholesky factor L is maintained via rank-1 updates.

Show that online SKA matches batch SKA's retrieval accuracy while
being O(1) per token.

No GPU required. Runs on CPU.
"""

import sys
sys.path.append("..")

import torch
import time
from shared.ska import SKAModule, robust_cholesky
from shared.eval_tasks import MQARDataset, PhonebookDataset
from shared.utils import evaluate, collate_fn


class OnlineSKAModule(torch.nn.Module):
    """
    Online SKA: maintains running sufficient statistics and updates
    the Cholesky factor incrementally.

    The batch SKA computes:
        G = sum_t z_t z_t^T           (Gram matrix)
        M = sum_t z_{t+1} z_t^T       (transition covariance)
        C_v = sum_t v_t z_t^T         (value readout)
        L = cholesky(G + lambda*I)
        A_w = M @ (G + lambda*I)^{-1}
        B_v = C_v @ (G + lambda*I)^{-1}

    All of G, M, C_v are sums of outer products, so they can be
    updated incrementally. L can be maintained via rank-1 Cholesky
    updates in O(r^2).
    """
    def __init__(self, d_model, n_heads, rank=24, head_dim=None, ridge_eps=1e-3):
        super().__init__()
        self.rank = rank
        self.ridge_eps = ridge_eps
        self.H = n_heads
        self.P = head_dim or (d_model // n_heads)

        self.key_proj = torch.nn.Linear(d_model, n_heads * rank, bias=False)
        self.query_proj = torch.nn.Linear(d_model, n_heads * rank, bias=False)
        self.value_proj = torch.nn.Linear(d_model, n_heads * self.P, bias=False)
        self.out_proj = torch.nn.Linear(n_heads * self.P, d_model, bias=False)

    def _rank1_cholesky_update(self, L, x):
        """
        Given lower-triangular L where L @ L^T = A, compute L' where
        L' @ L'^T = A + x @ x^T.

        This is the rank-1 Cholesky update algorithm.

        Args:
            L: (r, r) lower triangular
            x: (r,) vector

        Returns:
            L': (r, r) updated lower triangular

        TODO: Implement the rank-1 Cholesky update.
        Hint: This is a standard algorithm. For each column j=0..r-1:
            r_val = sqrt(L[j,j]^2 + x[j]^2)
            c = r_val / L[j,j]
            s = x[j] / L[j,j]
            L[j,j] = r_val
            For rows below j:
                L[j+1:, j] = (L[j+1:, j] + s * x[j+1:]) / c
                x[j+1:] = c * x[j+1:] - s * L[j+1:, j]
        """
        raise NotImplementedError("TODO: implement rank-1 Cholesky update")

    def process_token(self, state, h_t, h_prev=None):
        """
        Process a single token and update the running state.

        Args:
            state: dict with keys 'G', 'M', 'C_v', 'L', 'z_prev'
                   or None for the first token
            h_t: (1, d_model) hidden state for this token

        Returns:
            updated state dict

        TODO: Implement the incremental update:
        1. Project h_t to get z_t and v_t
        2. Update G += z_t @ z_t^T
        3. If z_prev exists, update M += z_t @ z_prev^T
        4. Update C_v += v_t @ z_t^T
        5. Update L via rank-1 Cholesky update
        6. Store z_prev = z_t
        """
        raise NotImplementedError("TODO: implement incremental token processing")

    def query(self, state, h_q):
        """
        Query the accumulated operator with a query token.

        Args:
            state: accumulated state from process_token calls
            h_q: (1, d_model) query hidden state

        Returns:
            (1, d_model) retrieval output

        TODO: Implement the query:
        1. Project h_q to get z_q
        2. Compute A_w from state M and L
        3. Compute B_v from state C_v and L
        4. Apply: output = B_v @ A_w^K @ L^{-1} @ z_q
        """
        raise NotImplementedError("TODO: implement query against accumulated state")


def compare_batch_vs_online():
    """
    Test that online SKA produces the same output as batch SKA.

    TODO: Complete this function:
    1. Create a small SKA and OnlineSKA with shared weights
    2. Generate random input sequence
    3. Run batch SKA on the full sequence
    4. Run OnlineSKA token by token, then query
    5. Compare outputs (should match to floating point precision)
    """
    torch.manual_seed(42)
    d_model, n_heads, rank = 64, 4, 16
    seq_len = 128

    batch_ska = SKAModule(d_model, n_heads, rank=rank)

    online_ska = OnlineSKAModule(d_model, n_heads, rank=rank)
    online_ska.key_proj.weight.data.copy_(batch_ska.key_proj.weight.data)
    online_ska.query_proj.weight.data.copy_(batch_ska.query_proj.weight.data)
    online_ska.value_proj.weight.data.copy_(batch_ska.value_proj.weight.data)
    online_ska.out_proj.weight.data.copy_(batch_ska.out_proj.weight.data)

    x = torch.randn(1, seq_len, d_model)

    # TODO: run batch SKA
    # TODO: run online SKA token by token
    # TODO: compare outputs and print max absolute difference

    print("TODO: implement comparison")


def benchmark_scaling():
    """
    Measure time per token for batch vs online SKA as context grows.

    TODO: Complete this function:
    1. For context lengths [128, 256, 512, 1024, 2048, 4096]:
       a. Time batch SKA on full sequence
       b. Time online SKA processing all tokens then querying
    2. Print table showing time per token for each method
    3. Batch should scale linearly, online should stay constant
    """
    print("TODO: implement scaling benchmark")


def eval_on_mqar():
    """
    Compare batch SKA vs online SKA on MQAR retrieval accuracy.

    TODO: Complete this function:
    1. Build a small model using batch SKA
    2. Build the same model using online SKA
    3. Train both on MQAR (or share weights)
    4. Evaluate retrieval accuracy -- should be identical
    """
    print("TODO: implement MQAR evaluation")


if __name__ == "__main__":
    compare_batch_vs_online()
    benchmark_scaling()
    eval_on_mqar()

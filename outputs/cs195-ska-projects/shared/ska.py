import math
import torch
import torch.nn as nn
from contextlib import nullcontext


def robust_cholesky(G, max_retries=4):
    G_sym = 0.5 * (G + G.transpose(-1, -2))
    for attempt in range(max_retries):
        try:
            return torch.linalg.cholesky(G_sym)
        except torch.linalg.LinAlgError:
            jitter = 1e-4 * (10 ** attempt)
            eye = torch.eye(G_sym.shape[-1], device=G_sym.device, dtype=G_sym.dtype)
            G_sym = G_sym + jitter * eye
    eigvals, eigvecs = torch.linalg.eigh(G_sym)
    eigvals = eigvals.clamp(min=1e-6)
    return torch.linalg.cholesky(
        eigvecs @ torch.diag_embed(eigvals) @ eigvecs.transpose(-1, -2))


def spectral_normalize_power_iter(A, n_iters=6):
    v = torch.ones(*A.shape[:-1], 1, device=A.device, dtype=A.dtype) / math.sqrt(A.shape[-1])
    with torch.no_grad():
        for _ in range(n_iters):
            Av = A @ v
            u = Av / Av.norm(dim=-2, keepdim=True).clamp(min=1e-8)
            Atu = A.transpose(-1, -2) @ u
            v = Atu / Atu.norm(dim=-2, keepdim=True).clamp(min=1e-8)
        sigma_max = (A @ v).norm(dim=-2, keepdim=False).squeeze(-1)
    scale = torch.clamp(sigma_max, min=1.0).unsqueeze(-1).unsqueeze(-1)
    return A / scale, sigma_max


def power_spectral_filter(A_w, w_q, power_K=2):
    A_filt = A_w
    for _ in range(power_K - 1):
        A_filt = A_filt @ A_w
    return A_filt @ w_q


class SKAModule(nn.Module):
    """
    Spectral Koopman Attention.

    Builds a Koopman operator from prefix tokens via ridge regression,
    then applies it to query positions for retrieval. Replaces standard
    dot-product attention with O(r^2) state instead of O(T) KV cache.

    Args:
        d_model: residual stream dimension
        n_heads: number of attention heads
        rank: dimension of the Koopman operator (controls state size)
        head_dim: value head dimension (defaults to d_model // n_heads)
        ridge_eps: ridge regression regularization
        scale: initial output scale
        power_K: number of power iterations for spectral filtering
    """
    def __init__(self, d_model, n_heads, rank=48, head_dim=None,
                 ridge_eps=1e-3, scale=1.5, power_K=2):
        super().__init__()
        self.rank = rank
        self.ridge_eps = ridge_eps
        self.power_K = power_K
        self.H = n_heads
        self.P = head_dim or (d_model // n_heads)
        self.d_model = d_model

        self.key_proj = nn.Linear(d_model, n_heads * rank, bias=False)
        self.query_proj = nn.Linear(d_model, n_heads * rank, bias=False)
        self.value_proj = nn.Linear(d_model, n_heads * self.P, bias=False)
        self.out_proj = nn.Linear(n_heads * self.P, d_model, bias=False)

        nn.init.orthogonal_(self.key_proj.weight)
        nn.init.orthogonal_(self.query_proj.weight)
        nn.init.xavier_uniform_(self.value_proj.weight)
        nn.init.zeros_(self.out_proj.weight)

        self.eta = nn.Parameter(torch.tensor(scale))
        self.ssn_gamma = nn.Parameter(torch.tensor(1.0))

    def forward(self, hidden_states, prefix_mask=None):
        B, T, _ = hidden_states.shape
        r = self.rank
        H, P = self.H, self.P

        z = self.key_proj(hidden_states).reshape(B, T, H, r)
        zq = self.query_proj(hidden_states).reshape(B, T, H, r)
        v = self.value_proj(hidden_states).reshape(B, T, H, P)

        ctx = torch.amp.autocast('cuda', enabled=False) if hidden_states.is_cuda \
              else nullcontext()

        with ctx:
            z_f = z.float()
            zq_f = zq.float()
            v_f = v.float()

            if prefix_mask is not None:
                m = prefix_mask.float().unsqueeze(-1).unsqueeze(-1)
            else:
                m = torch.ones(B, T, 1, 1, device=hidden_states.device, dtype=torch.float32)

            z_norms = z_f.norm(dim=-1, keepdim=True)
            max_norm = z_norms.max(dim=1, keepdim=True)[0].clamp(min=1e-6)
            z_f = z_f / max_norm
            zq_f = zq_f / max_norm

            z_m = z_f * m
            G = torch.einsum('bthr,bths->bhrs', z_m, z_m)

            if prefix_mask is not None:
                m_lag = (prefix_mask[:, :-1] * prefix_mask[:, 1:]).float().unsqueeze(-1).unsqueeze(-1)
            else:
                m_lag = torch.ones(B, T - 1, 1, 1, device=hidden_states.device, dtype=torch.float32)

            M_cov = torch.einsum('bthr,bths->bhrs',
                                 z_f[:, 1:] * m_lag,
                                 z_f[:, :-1] * m_lag)

            C_v = torch.einsum('bthp,bthr->bhpr', v_f * m, z_m)

            G_tilde = G + self.ridge_eps * torch.eye(r, device=G.device, dtype=G.dtype)
            L = robust_cholesky(G_tilde)

            Y = torch.linalg.solve_triangular(L, M_cov, upper=False)
            Aw_T = torch.linalg.solve_triangular(L, Y.transpose(-1, -2), upper=False)
            A_w = Aw_T.transpose(-1, -2)

            A_w, sigma_max = spectral_normalize_power_iter(A_w)
            gamma_safe = torch.clamp(self.ssn_gamma, min=1.0, max=1.5)
            A_w = A_w * gamma_safe

            B_v = torch.cholesky_solve(C_v.transpose(-1, -2), L).transpose(-1, -2)

            zq_perm = zq_f.permute(0, 2, 3, 1)
            w_q = torch.linalg.solve_triangular(L, zq_perm, upper=False)
            w_f = power_spectral_filter(A_w, w_q, self.power_K)
            z_f_out = L @ w_f
            y_hat = (B_v @ z_f_out).permute(0, 3, 1, 2)

        y_hat = self.eta * y_hat.to(hidden_states.dtype)
        output = self.out_proj(y_hat.reshape(B, T, H * P))
        return output

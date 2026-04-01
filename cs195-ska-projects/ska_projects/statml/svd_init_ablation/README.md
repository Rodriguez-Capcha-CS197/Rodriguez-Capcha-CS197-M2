# M1: SVD Initialization Ablation for SKA Surgery — Full Project Guide

## Project Classification: Math/Theory-Heavy

**Tier:** Intermediate | **GPU:** ~1 hour (one-time) | **Team Size:** 1–2 | **Total Duration:** 10 weeks (3 prep + 7 build)

**Dependencies:** None


## Project Summary

You are running an empirical study comparing 7 different weight initialization strategies for the SKA surgery — the process of replacing standard attention layers with Spectral Koopman Attention (SKA) modules. You extract real attention weights, initialize SKA modules with each strategy, analyze their mathematical properties, and measure how each affects training convergence. The final deliverable is a data-backed recommendation for the default initialization strategy. 

## Starting Requirements

Before you begin building anything, you need to be comfortable with the following concepts and tools. The 3 prep weeks are designed to get you there.

### Mathematical Prerequisites

- **Linear algebra fundamentals:** Matrix multiplication, transpose, inverse, rank, norms (Frobenius norm in particular). You should be able to explain what ||A||_F means and compute it.
- **Singular Value Decomposition (SVD):** You need to understand what U, Σ, V^T are, what the singular values represent geometrically (scaling factors along principal directions), and why truncated SVD is an optimal low-rank approximation (Eckart-Young theorem).
- **Condition number:** What κ(A) = σ_max / σ_min means, why large condition numbers cause numerical instability, and how ridge regularization (adding εI to a matrix) improves conditioning.
- **Gram matrices:** Given vectors z_1, ..., z_n, the Gram matrix G = Σ z_t z_t^T. Understanding why G is positive semi-definite, what its eigenvalues tell you, and how it connects to the Koopman operator construction.
- **PCA vs SVD:** PCA centers the data before computing the SVD. You should understand when centering matters and when it doesn't.
- **Non-negative Matrix Factorization (NMF):** A factorization A ≈ WH where W, H ≥ 0. You don't need deep theory — just understand the constraint and what it means for the factors.

### Programming Prerequisites

- Python with NumPy and SciPy for all matrix computations
- Matplotlib or Plotly for generating publication-quality plots
- Basic PyTorch: loading a pretrained model, extracting named parameters, saving/loading tensors
- Ability to write reproducible experiments (setting random seeds, organizing results into JSON)

### Codebase Familiarity

You will interact with a small number of files:
- `ska_agent/models/jamba_ska.py` — the `svd_init_ska_weights()` function you are studying
- `ska_agent/core/structures.py` — `SKAConfig` for rank, n_heads, head_dim parameters
- `ska_agent/utils/math_utils.py` — `SpectralUtils` for Gram matrix construction and condition number computation


## Prep Phase (Weeks 1–3)

The prep phase builds your mathematical and practical foundations. No project code is written yet — this is about understanding the theory well enough to design good experiments.

### Prep Week 1: Linear Algebra and SVD Foundations

**Goal:** Internalize SVD deeply enough to explain every initialization strategy.

**Study tasks:**
1. Review SVD decomposition. Work through the SVD of a small 3×4 matrix by hand (or with NumPy, checking each step). Verify that A = UΣV^T reconstructs exactly.
2. Implement truncated SVD: take the top-r singular values and corresponding vectors. Compute the reconstruction error ||A - A_r||_F and verify it equals sqrt(σ_{r+1}^2 + ... + σ_n^2) — this is the Eckart-Young theorem.
3. Understand the sqrt-Σ scaling used in the current codebase: if you set W_K = diag(√σ_i) · V[:r, :], what does this do geometrically? The sqrt distributes the "energy" (singular values) symmetrically between keys and queries. Write a short paragraph explaining why this might be preferable to putting all the energy in keys (full-Σ) or none (no scaling).
4. Compute condition numbers of random matrices and observe how adding εI changes them.

**Deliverable:** A 1-page summary of SVD, truncated SVD, and the Eckart-Young theorem, written in your own words. Include a worked numerical example.

### Prep Week 2: Gram Matrices and the Koopman Operator

**Goal:** Understand what the SKA module computes and why initialization matters.

**Study tasks:**
1. Read the Koopman operator construction path: G = Σ z_t z_t^T + εI (Gram matrix with ridge), M = Σ z_t z_{t-1}^T (transition matrix), L = cholesky(G), A_w = L^{-1} M L^{-T} (whitened operator). Work through this on a small example (r=4, T=10 random vectors).
2. Understand why the condition number of G matters: if κ(G) is huge, the Cholesky factorization amplifies numerical errors, and A_w becomes unreliable. Verify this by constructing a near-singular G and observing the effect on A_w.
3. Connect initialization to Gram conditioning: the keys z_t = x · W_K^T. If W_K has near-zero singular values, some directions in key space are collapsed, making G ill-conditioned. If W_K has very large singular values, the keys may overflow the dynamic range. The sqrt-Σ scaling is a compromise.
4. Run the existing `svd_init_ska_weights()` on extracted attention weights (follow the starter code to extract weights from Qwen2.5-1.5B). Examine the singular value distribution of the resulting SKA key projection.

**Deliverable:** A Jupyter notebook that constructs G and A_w from random keys, computes condition numbers, and shows how different key distributions affect stability.

### Prep Week 3: Experimental Design and Baselines

**Goal:** Plan your full experimental pipeline before writing it.

**Study tasks:**
1. For each of the 7 strategies, write down exactly what it computes mathematically and what properties you expect (condition number, reconstruction error, isotropy of resulting keys).
2. Design your analysis metrics: condition number of the resulting Gram matrix, spectral gap (σ_1 - σ_2 of the operator), reconstruction error vs the original weight matrix, and convergence speed under training.
3. Plan your experimental protocol: how many random seeds, what proxy loss function for the small-scale training comparison, what rank values to test, what constitutes a statistically significant difference.
4. Set up your project directory structure: scripts for weight extraction, initialization, analysis, training, and plotting. Write skeleton code for each.

**Deliverable:** A written experimental plan (1–2 pages) listing hypotheses, metrics, number of runs, and expected outcomes. Plus a project directory with skeleton scripts.

## Build Phase (Weeks 4–10)

### Build Week 1 (Week 4): Weight Extraction and All 7 Strategies

**Tasks:**
- Run the weight extraction script on Qwen2.5-1.5B-Instruct. Save attention weights to disk. This is your one-time GPU use.
- Implement all 7 initialization strategies. Verify each produces the correct output shape (n_heads × rank, d_model).
- For each strategy, compute and print: output shape, Frobenius norm, min/max singular values, and check for NaN/Inf.

**Expected output:** A pickle file of extracted weights. A module with 7 functions that each take a weight matrix and rank, and return an initialized SKA key projection. A summary table of basic statistics for all 7 strategies applied to 3 different attention layers.

### Build Week 2 (Week 5): Spectral Analysis Pipeline

**Tasks:**
- For each strategy, generate keys from random input, build the Gram matrix, and compute: mean/max/std condition number across heads, spectral gap of the operator, and reconstruction error against the original attention weights.
- Produce a summary table with 7 rows (strategies) and columns for each metric.
- Generate singular value distribution plots for each strategy (bar charts showing the singular values of the initialized W_K).

**Expected output:** A results table comparing all 7 strategies on spectral health metrics. Singular value distribution plots. The SVD-based strategies should show smoother, more controlled spectra than random baselines.

### Build Week 3 (Week 6): Small-Scale Training Comparison

**Tasks:**
- Create small SKA modules (d_model=256, rank=16, n_heads=4) initialized with each strategy.
- Train each for 2000 steps on synthetic data with a proxy LM loss (mean squared error on output).
- Plot all 7 training curves on the same axes. Run with 3 different random seeds and compute mean/std of loss at steps 100, 500, 1000, 2000.

**Expected output:** Training convergence plots with error bars. SVD-based methods should converge faster than random baselines. A table of loss values at 4 checkpoints for all 7 strategies.

### Build Week 4 (Week 7): Rank Ablation

**Tasks:**
- Repeat the spectral analysis and training comparison for ranks r = 8, 16, 32, 64.
- Produce 4 condition number heatmaps (one per rank) showing all 7 strategies.
- Determine whether the optimal strategy changes with rank.

**Expected output:** Heatmaps and learning curves across 4 rank settings. Analysis of whether the ranking of strategies is stable or rank-dependent.

### Build Week 5 (Week 8): Reconstruction Error Deep Dive

**Tasks:**
- For each head and each strategy, compute the relative reconstruction error: ||K_h_original - expand(W_K_SKA_h)||_F / ||K_h_original||_F.
- Compare against the Eckart-Young theoretical lower bound (which only SVD methods can approach).
- Produce bar charts of reconstruction error for all 7 strategies, with the theoretical bound marked.

**Expected output:** Reconstruction error comparison. SVD methods should match or approach the Eckart-Young bound. Non-SVD methods (random, NMF) should be strictly worse.

### Build Week 6 (Week 9): Statistical Validation

**Tasks:**
- For the top 3 strategies from your analysis, re-run the full training comparison with 5 random seeds each.
- Compute confidence intervals on all metrics. Run paired t-tests to determine which differences are statistically significant (p < 0.05).
- Produce final comparison plots with proper error bars and significance annotations.

**Expected output:** Statistically validated ranking of initialization strategies with confidence intervals and p-values.

### Build Week 7 (Week 10): Report and Recommendations

**Tasks:**
- Write a 3–4 page report summarizing all findings. Structure: Introduction (what is SKA initialization and why it matters), Methods (7 strategies and metrics), Results (spectral analysis, training convergence, rank ablation, reconstruction error), Discussion (which strategy wins and why), Recommendation (specific strategy with rank-dependent guidance).
- Include all key figures: singular value distributions, training curves, heatmaps, reconstruction error bars, Pareto plots if applicable.
- Clean up all code into a reproducible pipeline.

**Expected output:** A complete written report with a clear recommendation backed by data. The recommendation should specify which strategy to use as default, and under what conditions (rank, model size) alternatives might be preferred.


## What "Done" Looks Like

By the end of this project, you will have:

1. A tested implementation of 7 initialization strategies for SKA surgery
2. Comprehensive spectral health analysis (condition numbers, spectral gaps, singular value distributions) across all strategies and multiple ranks
3. Training convergence comparisons with statistical validation (multiple seeds, error bars, significance tests)
4. A reconstruction error analysis comparing against theoretical lower bounds
5. A written report with a specific, data-backed recommendation for the default initialization strategy

The recommendation should be concrete: "Use SVD with sqrt-Σ scaling (Strategy 3) as the default for ranks 8–64. At rank ≥ 128, consider Strategy X because..." — not a vague "it depends."

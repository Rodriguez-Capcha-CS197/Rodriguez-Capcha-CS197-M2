# M3: Spectral and Orthogonal Regularization Interaction — Full Project Guide

## Project Classification: Math/Theory-Heavy

**Tier:** Advanced | **GPU:** ~2 hours | **Team Size:** 1 | **Total Duration:** 10 weeks (3 prep + 7 build)

**Dependencies:** None

---

## Project Summary

You are running a systematic empirical study of how two regularization losses — spectral regularization (λ_spec) and orthogonal regularization (λ_ortho) — interact during SKA module training. You run a 25-point grid sweep over their strengths, track operator health metrics throughout training, visualize the joint landscape, identify Pareto-optimal settings, and characterize corner-case behavior. The deliverable is a recommendation for default regularization parameters backed by rigorous analysis.

---

## Motivation: Why This Matters for SKA-Agent

The Koopman operator at the heart of SKA-Agent is a powerful but delicate object. It compresses temporal dynamics (how key representations evolve across a prefix) into a fixed-size matrix, and the power spectral filter A_w^K amplifies directions with consistent signal while suppressing noise. But power iteration is inherently unstable if the operator isn't properly constrained: if the spectral radius exceeds 1, repeated application of A_w explodes; if the Gram matrix is ill-conditioned, the whitening step (L^{-1} M L^{-T}) amplifies numerical errors into the operator itself.

This is where the two regularizers come in, and understanding their interaction is essential for the whole system. Spectral regularization (penalizing ||A_w||_F²) directly controls the operator's strength — it prevents the power filter from amplifying too aggressively, but if set too high, it makes the operator a no-op and the SKA module contributes nothing. Orthogonal regularization (penalizing ||W_K^T W_K - I||) controls the input conditioning — it ensures the key projections preserve geometry so the Gram matrix stays well-conditioned, but if set too high, it over-constrains the projections and limits what the operator can learn.

The tension between these regularizers is fundamental to SKA-Agent's architecture. In the shared spectral memory protocol, agents communicate through operators — a retriever agent writes keys that build one operator, a coordinator's reasoning states build another, and the multi-head Koopman module runs 4 parallel operators per head with different slot assignments. Each of these operators needs to be both stable (well-conditioned) and expressive (not a no-op). Your grid sweep directly maps the landscape of this tradeoff and determines the operating point for every Koopman operator in the system.

---

## Starting Requirements

### Mathematical Prerequisites

- **Regularization in machine learning:** You need to understand why we add penalty terms to loss functions — to prevent overfitting, enforce structural properties, or improve numerical stability. You should know L2 regularization (weight decay) as a baseline, and understand that different penalties encourage different properties.
- **Singular values and spectral radius:** The spectral radius ρ(A) = max|σ_i(A)| controls how the operator behaves under repeated application (power iteration). If ρ > 1, repeated application amplifies; if ρ < 1, it contracts. The spectral regularization loss L_spec = λ_spec · Σ σ_i(A_w)² is the squared Frobenius norm, which penalizes all singular values (not just the largest).
- **Orthogonality and isometry:** The orthogonal regularization loss L_ortho = λ_ortho · ||W_K^T W_K - I||_F² penalizes deviation from isometry (distance-preserving). When W_K is orthogonal, it preserves dot products between inputs, which means the Gram matrix G = Σ (W_K x_t)(W_K x_t)^T inherits the conditioning of the input covariance. Understand why this is desirable.
- **Condition number dynamics:** How condition number changes during training. Without regularization, the condition number of G can grow unboundedly as keys become more correlated. Orthogonal reg prevents this by keeping W_K well-conditioned.
- **Pareto optimality:** A configuration is Pareto-optimal if no other configuration is strictly better on all objectives simultaneously. You need to understand multi-objective optimization at a conceptual level — "quality" and "stability" are competing objectives that cannot both be maximized.
- **Statistical significance:** p-values, t-tests, confidence intervals. You need to distinguish real differences from noise when comparing configurations across random seeds.

### Programming Prerequisites

- PyTorch: training loops, optimizers, computing gradients, model.parameters()
- NumPy/SciPy for SVD computation and statistical tests
- Matplotlib/Plotly for heatmaps, training curves, and Pareto frontier plots
- JSON for saving experimental results
- Ability to run 25+ training configurations programmatically (scripting grid sweeps)

### Codebase Familiarity

- `ska_agent/training/trainers.py` — `SpectralRegularization.forward()` and `OrthogonalRegularization.forward()`. Read these carefully to understand exactly what they compute.
- `ska_agent/core/ska_module.py` — `SKAModule` and `get_operator_stats()`, which returns condition number, spectral radius, and gate value.
- `ska_agent/core/structures.py` — `SKAConfig` for module dimensions.

---

## Prep Phase (Weeks 1–3)

### Prep Week 1: Spectral Theory for SKA Operators

**Goal:** Understand what the two regularizers do mathematically and why they might interact.

**Study tasks:**
1. Work through the Koopman operator construction: G → L (Cholesky) → A_w = L^{-1} M L^{-T}. Understand that A_w is a "whitened" operator — its eigenvalues equal those of the natural operator A = M G^{-1}, but A_w is better conditioned for numerical computation.
2. Understand spectral regularization: L_spec penalizes ||A_w||_F². Since ||A_w||_F² = Σ σ_i², this shrinks all singular values toward zero. Extreme case: if λ_spec is very large, A_w ≈ 0 and the operator does nothing (the SKA module becomes a no-op).
3. Understand orthogonal regularization: L_ortho penalizes ||W_K^T W_K - I||_F². This pushes W_K toward an orthogonal matrix (W_K^T W_K = I means W_K preserves distances). When W_K is orthogonal, the keys z = W_K x preserve the conditioning of the input x, so G is well-conditioned.
4. Think about the interaction: spectral reg controls the operator's output strength, orthogonal reg controls the input conditioning. If both are strong, the operator is well-conditioned but weak. If both are zero, the operator is unconstrained and potentially ill-conditioned. The interesting question is the tradeoff surface.

**Deliverable:** A 1-page document explaining both regularizers in your own words, with a prediction of what happens at the 4 corners of the (λ_spec, λ_ortho) grid: (0,0), (high,0), (0,high), (high,high).

### Prep Week 2: Training Dynamics and Health Metrics

**Goal:** Learn to monitor operator health during training.

**Study tasks:**
1. Run a single training configuration (the starter code) for 2000 steps. At every 100 steps, record: total loss, condition number, spectral radius, gate value, and ||W_K^T W_K - I||_F.
2. Plot all 5 metrics vs training step. Understand what a "healthy" trajectory looks like: loss decreasing, condition number stable (not exploding), spectral radius below 1 (contractive operator), gate value moving from 0.5 toward its learned value.
3. Run the same configuration without any regularization (λ_spec=0, λ_ortho=0). Compare the trajectories. Without regularization, condition number should grow and spectral radius may exceed 1.
4. Understand the gate value: the SKA module has a learned gate that interpolates between the SKA output and a residual connection. If the gate goes to 0, the SKA module has effectively shut itself off (a sign of over-regularization or training failure).

**Deliverable:** A notebook with 5-panel training trajectory plots for the default config and the unregularized config, with annotations explaining what each metric tells you.

### Prep Week 3: Experimental Design for Grid Sweeps

**Goal:** Plan the full 25-point experiment.

**Study tasks:**
1. Choose the grid: λ_spec ∈ {0, 0.001, 0.01, 0.1, 1.0} and λ_ortho ∈ {0, 0.001, 0.01, 0.1, 1.0}. This gives 25 configurations. Estimate total training time (25 × 2000 steps ≈ 50K steps total, roughly 1–2 hours on a single GPU).
2. Decide on summary metrics: for each configuration, what single numbers will you put in the heatmap? Candidates: final loss, final condition number, final spectral radius, final gate value, final orthogonality residual. You'll make one heatmap per metric.
3. Plan the Pareto analysis: define "quality" as negative final loss (lower loss = higher quality) and "stability" as inverse condition number (lower κ = more stable). Each of the 25 configurations becomes a point in (quality, stability) space.
4. Plan the corner case analysis: at each of the 4 extreme corners, what specific behavior do you expect? Write down predictions to test.

**Deliverable:** A written experimental plan with grid values, summary metrics, Pareto objective definitions, and corner-case predictions.

---

## Build Phase (Weeks 4–10)

### Build Week 1 (Week 4): Single Configuration Validation + Grid Sweep

**Tasks:**
- Verify the training pipeline on 2–3 configurations: default (0.01, 0.01), no reg (0, 0), and heavy reg (0.1, 0.1). Make sure `get_operator_stats()` returns sensible values and the training loop doesn't crash.
- Run the full 25-point grid sweep. Save all metrics (at every 100 steps) for all 25 configs to a JSON file.

**Expected output:** A JSON file with complete training trajectories for all 25 configurations. No NaN or Inf values. All 25 runs completed.

### Build Week 2 (Week 5): Heatmap Visualizations

**Tasks:**
- For each of 5 final metrics (loss, condition number, spectral radius, gate value, orthogonality residual), create a 5×5 heatmap over the (λ_spec, λ_ortho) grid.
- Use log scale for condition number (it spans orders of magnitude). Use consistent color schemes.
- Annotate the heatmaps with the actual numeric values in each cell.

**Expected output:** 5 publication-quality heatmaps. You should see clear patterns: high λ_spec corner has low spectral radius, high λ_ortho corner has low condition number, (0,0) corner has the lowest loss but worst stability.

### Build Week 3 (Week 6): Training Curve Comparisons + Pareto Analysis

**Tasks:**
- Select 4–5 representative configurations: the 4 corners + the best-performing config. Plot their training curves (loss, κ, ρ) on the same axes for direct comparison.
- Compute the Pareto frontier in (quality, stability) space. Plot all 25 points, mark the Pareto-optimal ones, and identify dominated configurations.

**Expected output:** Overlaid training curves showing clear behavioral differences. Pareto frontier with 3–5 non-dominated configurations identified. Dominated configurations marked.

### Build Week 4 (Week 7): Sensitivity Analysis with Multiple Seeds

**Tasks:**
- For the 3–5 Pareto-optimal configurations, re-run each with 3 different random seeds.
- Compute mean and standard deviation of final metrics across seeds.
- Run paired t-tests between neighboring Pareto-optimal configs to determine which differences are statistically significant (p < 0.05).
- Add error bars to the Pareto frontier plot.

**Expected output:** Error bars on all metrics for Pareto-optimal configs. At least one pair of configurations with a statistically significant difference.

### Build Week 5 (Week 8): Corner Case Analysis

**Tasks:**
- **(0, 0) — No regularization:** Track condition number over training. Identify when/if it explodes. Measure how many steps until the operator becomes numerically unusable (κ > 10^6).
- **(1.0, 1.0) — Maximum regularization:** Measure how much the SKA module contributes to the output. Check if the gate value goes to 0 (meaning the module has shut itself off).
- **(1.0, 0) — Spectral only:** Does the operator collapse to zero? Measure ||A_w||_F over training.
- **(0, 1.0) — Orthogonal only:** Is the condition number controlled? Does loss converge?

**Expected output:** A narrative description of each corner's behavior with 2–3 supporting plots per corner.

### Build Week 6 (Week 9): Robustness and Additional Analysis

**Tasks:**
- Test whether the results change at different ranks (r=32 vs r=64) or different d_model values.
- If time permits, explore whether a warm-start schedule (starting with high regularization and decaying) outperforms fixed regularization.
- Compile all figures and tables into a coherent narrative.

**Expected output:** Robustness check across at least one alternative setting. All figures finalized.

### Build Week 7 (Week 10): Written Report and Recommendations

**Tasks:**
- Write a 3–4 page report with: Introduction (what are the regularizers and why do they matter), Methods (grid sweep setup), Results (heatmaps, Pareto frontier, corner analysis), Discussion (interaction effects, when each regularizer dominates), Recommendation (specific default values with justification).
- The recommendation should be concrete: "Use λ_spec=0.01, λ_ortho=0.01 as the default. Increase λ_ortho to 0.1 if condition number exceeds 10^4 during training."
- Clean up all code.

**Expected output:** A complete report with clear recommendation. Code that reproduces all experiments.

---

## What "Done" Looks Like

1. A completed 25-point grid sweep with all training trajectories saved
2. Five heatmaps showing how each health metric varies across the grid
3. A Pareto frontier with sensitivity analysis (error bars, statistical tests)
4. Detailed corner case analysis explaining extreme behaviors
5. A written report with a specific, justified recommendation for default regularization parameters

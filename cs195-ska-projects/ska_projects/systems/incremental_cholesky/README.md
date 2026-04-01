# S1: Incremental Cholesky Operator Updates — Full Project Guide

## Project Classification: Math/Algorithms-Heavy

**Tier:** Advanced | **GPU:** None | **Team Size:** 1–2 | **Total Duration:** 10 weeks (3 prep + 7 build)

**Dependencies:** None. Pure NumPy/SciPy, no ML models needed.

---

## Project Summary

You are building an O(r²)-per-step incremental update for the Koopman operator in SharedSpectralMemory, replacing the current O(r³) from-scratch rebuild. This involves a rank-1 Cholesky update algorithm, incremental operator reconstruction via Givens rotation tracking, numerical stability monitoring, and a refactorization schedule. This is a numerical linear algebra project — you're optimizing a core mathematical routine for streaming settings.

---

## Motivation: Why This Matters for SKA-Agent

The shared spectral memory is SKA-Agent's central communication mechanism. Instead of agents passing thousands of tokens of text to each other (growing KV cache by ~32MB per handoff), an agent writes key vectors that update a rank-r operator (~32KB), and other agents query it through the power spectral filter. This is what makes multi-agent collaboration feasible without exploding memory costs — a retriever can share what it found with a reasoner through a 64×64 matrix instead of a 2,000-token context dump.

But every time an agent writes a new key, the operator needs to be rebuilt: the Gram matrix G gets a rank-1 update (G → G + zz^T), the transition matrix M gets updated, and the whitened operator A_w = L^{-1} M L^{-T} must be recomputed from the new Cholesky factor. The current implementation sets a `_stale` flag on write and rebuilds everything from scratch on the next read — a full Cholesky factorization at O(r³) plus two rounds of triangular solves. At r=64 this takes microseconds, but in a streaming multi-agent setting where keys arrive at every token and multiple agents are reading and writing concurrently, rebuilding from scratch becomes the bottleneck.

Your incremental update reduces this to O(r²) per key arrival by exploiting the rank-1 structure of the Gram update. The Cholesky factor can be updated with a sequence of Givens rotations, and the same rotations can be propagated to the cached whitened operator without re-solving from scratch. This is the difference between the spectral memory being a practical communication channel (updated incrementally in the inner loop) and a batch-only summary (rebuilt periodically in the outer loop). At r=256 or higher, the speedup becomes essential for real-time multi-agent operation.

---

## Starting Requirements

### Mathematical Prerequisites — This Is a Serious Math Project

- **Cholesky factorization:** Given a symmetric positive definite matrix G, there exists a unique lower triangular matrix L such that G = LL^T. You must understand the algorithm, why it requires positive definiteness, and what goes wrong numerically when G is nearly singular.
- **Rank-1 updates:** When G becomes G + zz^T, the new Cholesky factor L' can be computed from L in O(r²) using a sequence of Givens rotations. This is the core algorithm you'll implement. You must understand it deeply, not just copy it.
- **Givens rotations:** A Givens rotation G(i,j,θ) zeroes out one element of a vector by rotating in the (i,j) plane. Given (a, b), the rotation finds (c, s) such that c·a + s·b = √(a² + b²) and -s·a + c·b = 0. You should be able to derive this and apply it to a matrix.
- **The key insight for A_w update:** A_w = L^{-1} M L^{-T}. When both L and M change, updating A_w naively requires O(r³) triangular solves. The optimization is to maintain U = L^{-1}M and apply the Givens rotations from the Cholesky update to U. Each Givens rotation modifies two rows at O(r) cost; r rotations give O(r²) total. You must work through this derivation carefully.
- **Spectral normalization:** After updating A_w, it must be normalized so σ_max(A_w) ≤ γ. Power iteration finds σ_max. Warm-starting power iteration from the previous dominant singular vector saves iterations.
- **Numerical stability:** Floating-point errors accumulate over many incremental updates. You need to understand relative error, when drift becomes problematic, and why periodic full refactorization is necessary.

**Recommended reading:** Golub and Van Loan, "Matrix Computations," specifically Section 6.5.4 (rank-1 Cholesky updates) and Section 5.1 (Givens rotations). If you don't have access to the textbook, the Wikipedia articles on Cholesky decomposition and Givens rotations are sufficient starting points, but you'll need to derive the A_w update formulation yourself.

### Programming Prerequisites

- Python with NumPy and SciPy (no PyTorch needed)
- Strong comfort with array indexing and in-place operations
- Experience with numerical testing (comparing floating-point results with tolerance)
- Performance benchmarking (timing code, profiling, understanding algorithmic complexity)

### Codebase Familiarity

- `ska_agent/shared_memory/spectral_memory.py` — `SharedSpectralMemory.write()` sets `_stale=True`, and the `operator` property triggers a full rebuild. You're making the rebuild incremental.
- `ska_agent/utils/math_utils.py` — `SpectralUtils.whiten_operator()` and `spectral_normalize()` — the from-scratch approach you're replacing.

---

## Prep Phase (Weeks 1–3)

### Prep Week 1: Cholesky Factorization and Rank-1 Updates

**Goal:** Understand and implement the rank-1 Cholesky update from first principles.

**Study tasks:**
1. Implement Cholesky factorization from scratch (not using `np.linalg.cholesky`). The algorithm processes one column at a time: L[j,j] = √(G[j,j] - Σ_{k<j} L[j,k]²), then L[i,j] = (G[i,j] - Σ_{k<j} L[i,k]L[j,k]) / L[j,j]. Run it on random SPD matrices and verify against NumPy.
2. Study the rank-1 update algorithm. Given L such that LL^T = G, and a new vector z, compute L' such that L'L'^T = G + zz^T. The algorithm processes diagonal entries i = 0, ..., r-1, computing Givens parameters (c, s) at each step that absorb z[i] into L[i,i], then propagating to subsequent rows.
3. Implement the rank-1 update (starter code is provided). Verify it against full Cholesky: compute G + zz^T, factor from scratch, and check that L'L'^T matches to machine precision (~1e-12 relative error).
4. Run the verification at r = 32, 64, 128. All should pass.

**Deliverable:** A working `cholesky_rank1_update(L, z)` function that passes verification tests at multiple ranks. A written explanation of the algorithm in your own words, with a diagram showing how the Givens rotation at step i modifies L and z.

### Prep Week 2: Givens Rotations Applied to External Matrices

**Goal:** Understand how to propagate Cholesky update rotations to a separate matrix — this is the intellectual core of the project.

**Study tasks:**
1. Understand what a Givens rotation does to a matrix. If you apply G(i,j,θ) to the left of a matrix M, it mixes rows i and j: row_i' = c·row_i + s·row_j, row_j' = -s·row_i + c·row_j. Each such rotation costs O(r) (it only touches two rows).
2. The Cholesky update computes a sequence of r Givens rotations that transform [L | z] into [L' | 0]. These same rotations, applied to the cached matrix U = L^{-1}M, will produce a matrix that is "almost" L'^{-1}M (plus a rank-1 correction from the M update).
3. Work through the math carefully: L' = Q₁Q₂...Qᵣ · L (where Qᵢ are the Givens rotation matrices). Therefore L'^{-1} = L^{-1} · Qᵣ^T...Q₁^T. So L'^{-1}M = L^{-1}Qᵣ^T...Q₁^T M = (apply rotations to U^T, then transpose). Work this out for a small example (r=4).
4. Verify your understanding: apply the rotations from a Cholesky update to an identity matrix, and check that the result matches L'^{-1}L (the transformation between old and new Cholesky factors).

**Deliverable:** A working function that applies a sequence of Givens rotations to an arbitrary matrix. Verification test showing it correctly relates L'^{-1} to L^{-1}.

### Prep Week 3: Full Incremental Update Design

**Goal:** Design the complete incremental A_w update algorithm on paper before implementing it.

**Study tasks:**
1. Write out the full update derivation:
   - New Gram: G' = G + zz^T → L' via rank-1 Cholesky update (saves rotations)
   - New transition: M' = M + z·w^T (where w = z_{t-1})
   - Cached U = L^{-1}M → update U to U' = L'^{-1}M' = L'^{-1}(M + zw^T) = L'^{-1}M + (L'^{-1}z)(w^T)
   - The L'^{-1}M term: apply inverse Givens rotations to U, giving L'^{-1}M in O(r²)
   - The (L'^{-1}z) term: forward-solve L'v = z in O(r²), then add the rank-1 outer product vw^T in O(r²)
   - Finally: A_w' = U' · L'^{-T}, which is another round of Givens applications, also O(r²)
2. Identify the memory layout: you need to store L (lower triangular, r×r), M (r×r), U = L^{-1}M (r×r), and the current A_w (r×r). Total: 4r² floats.
3. Plan the spectral normalization: after updating A_w', compute σ_max via power iteration. Warm-start from the previous dominant singular vector (should converge in 2–3 iterations instead of 10+).
4. Plan the refactorization schedule: every N updates, do a full from-scratch computation and reset accumulated error. Choose N based on your stability analysis.

**Deliverable:** A complete written algorithm specification (pseudocode with complexity annotations) for the full incremental update. This document is your implementation blueprint.

---

## Build Phase (Weeks 4–10)

### Build Week 1 (Week 4): Givens Application and Core Infrastructure

**Tasks:**
- Implement the Givens rotation application to a separate matrix (from Prep Week 2, now production-quality).
- Implement the `IncrementalOperatorBuilder` class skeleton with `update()` and `_full_refactorize()`.
- Implement the U = L^{-1}M cache: after a full refactorization, compute U by forward-solving L·U = M.

**Expected output:** The builder class with full refactorization working (a correct but slow baseline), plus the Givens application helper verified against matrix multiplication.

### Build Week 2 (Week 5): Incremental A_w Update

**Tasks:**
- Implement the full incremental A_w update in the `update()` method: Cholesky update, Givens-based U update, rank-1 correction, and Givens-based A_w computation.
- Run the correctness test: 500 incremental updates at r=64, comparing against from-scratch at every step.
- Target: max relative error below 1e-10 over all 500 steps.

**Expected output:** `test_incremental_vs_full()` passes. The incremental path produces numerically identical results to the from-scratch path.

### Build Week 3 (Week 6): B_v Update and Spectral Normalization

**Tasks:**
- Implement incremental B_v update (value readout): B_v = C_v G^{-1}. After the Cholesky update, G^{-1} can be updated via the Woodbury identity or by solving with the new L.
- Implement spectral normalization with warm-started power iteration. Track the number of power iteration steps needed (should decrease over time as the dominant direction stabilizes).
- Run full end-to-end correctness test including B_v.

**Expected output:** Complete `IncrementalOperatorBuilder` that produces operators matching from-scratch to 1e-10. Power iteration convergence tracked.

### Build Week 4 (Week 7): Numerical Stability Analysis

**Tasks:**
- Run 10,000 incremental updates. Track Cholesky residual (||LL^T - G||_F / ||G||_F) and A_w relative error at every step.
- Run 100,000 updates. Characterize when drift becomes detectable (residual exceeds 1e-10, 1e-8, 1e-6).
- Analyze the drift pattern: does error grow linearly, as √n, or exponentially?

**Expected output:** Stability plots showing residual vs update count. Clear characterization of the drift rate. A recommendation for maximum updates between refactorizations.

### Build Week 5 (Week 8): Refactorization Scheduling

**Tasks:**
- Implement two refactorization strategies: (1) fixed interval (every N steps), (2) adaptive (refactorize when residual exceeds threshold).
- Measure amortized cost for each strategy: total wall-clock time divided by number of updates, including the occasional full refactorization.
- Find the optimal refactorization interval that keeps residual below 1e-8 with minimal overhead.

**Expected output:** Comparison of fixed vs adaptive refactorization. With fixed interval every 1000 steps, amortized cost should be within 5% of pure incremental (the occasional O(r³) refactorization is amortized over 1000 O(r²) updates).

### Build Week 6 (Week 9): Performance Benchmarks

**Tasks:**
- Benchmark at r = 32, 64, 128, 256, 512 for 1000 steps each.
- Measure from-scratch time vs incremental time at each rank.
- Perform roofline analysis: the theoretical speedup is O(r³)/O(r²) = O(r), but constant factors matter. At what rank does the incremental approach become faster?

**Expected output:** Benchmark table showing speedup at each rank. At r=256, expect at least 2× speedup. At r=64, the speedup may be modest due to overhead constants. Clear documentation of when incremental updates are worthwhile.

### Build Week 7 (Week 10): Integration and Report

**Tasks:**
- Integrate into `SharedSpectralMemory` with a `use_incremental=True` config flag. When enabled, `write()` calls the incremental builder instead of setting `_stale=True`.
- Run an integration test: 200-write sequence through the actual `write()/read()` API, comparing incremental vs from-scratch outputs.
- Write a 2–3 page report covering: the algorithm, correctness proof sketch, stability analysis, benchmark results, and when to use incremental vs from-scratch.

**Expected output:** Working integration with config flag. Integration test passes. Written report with algorithm description and performance guidance.

---

## What "Done" Looks Like

1. A correct O(r²)-per-step incremental update for the Koopman operator
2. Numerical stability analysis showing drift characteristics over 10K+ updates
3. Refactorization scheduling that maintains accuracy with minimal overhead
4. Performance benchmarks proving speedup at r ≥ 128
5. Integration into SharedSpectralMemory with a clean API
6. Written report explaining the algorithm, its limitations, and when to use it

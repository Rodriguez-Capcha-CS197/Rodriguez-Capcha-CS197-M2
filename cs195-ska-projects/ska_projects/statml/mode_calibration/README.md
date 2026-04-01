# M4: Mode Selector Calibration + Calibrated Routing — Full Project Guide

## Project Classification: Math/ML-Hybrid

**Tier:** Intermediate | **GPU:** Minimal | **Team Size:** 1 | **Total Duration:** 10 weeks (3 prep + 7 build)

**Dependencies:** Requires trained ModeSelector checkpoint from S5 Router Training (delivered at S5's week 3, which aligns with your build week 1). You use the prep phase and early build weeks to prepare the calibration infrastructure, then apply it to the trained model.

---

## Project Summary

You are studying whether the ModeSelector's softmax probabilities are trustworthy. If the model says "95% LOOKUP," is it right 95% of the time? If not, you apply post-hoc calibration methods (temperature scaling, Platt scaling, histogram binning) to fix the probabilities, then build an exploration policy that uses calibrated confidence to decide when to try alternative modes instead of blindly trusting the top prediction.

---

## Motivation: Why This Matters for SKA-Agent

The router classifies each query into one of four collaboration modes — LOOKUP, MULTI_DOC, COMPUTE, MULTI_STEP — and each mode activates a different DAG template that determines which specialists run, in what order, and how they share information through the spectral memory. Choosing the wrong mode doesn't just degrade quality slightly; it activates an entirely wrong workflow. A COMPUTE query misclassified as LOOKUP will never invoke the code executor, and no amount of retrieval can compensate.

This is why calibration matters more here than in a typical classifier. The ModeSelector outputs softmax probabilities, but neural networks are notoriously overconfident — it might say "95% LOOKUP" when it's only right 60% of the time for that confidence level. If the router trusts this blindly, it never explores alternatives. But if you calibrate the probabilities so they're trustworthy, a prediction of "55% LOOKUP, 35% MULTI_DOC" becomes a genuine signal that the query is ambiguous and might benefit from trying both workflows.

The exploration policy you build is a form of uncertainty-aware routing. When the calibrated confidence is low, the router tries the top two modes and picks the better result. This is feasible because SKA-Agent's specialists are fast (the retriever uses fixed-size operators, the code executor runs in a sandbox, the reasoner uses a small coordinator model) — trying two modes doesn't blow the budget. Your calibration work makes the router's uncertainty estimates reliable enough to drive this exploration efficiently.

---

## Starting Requirements

### Mathematical Prerequisites

- **Probability calibration:** A classifier is calibrated if, among all predictions where it says "80% class k," exactly 80% are truly class k. This is a stronger requirement than accuracy — a model can be accurate but poorly calibrated (e.g., always predicting 99% for the correct class when it's only right 85% of the time).
- **Expected Calibration Error (ECE):** Bin predictions by confidence level, compute the gap between average confidence and average accuracy in each bin, and take the weighted average. You need to understand this metric well enough to implement it and interpret it.
- **Reliability diagrams:** Plot average accuracy vs average confidence for each bin. A perfectly calibrated model lies on the diagonal. Points above the diagonal mean the model is underconfident; below means overconfident.
- **Temperature scaling:** Divide logits by a temperature T before softmax. T > 1 softens the distribution (reduces overconfidence). T < 1 sharpens it. The optimal T is found by minimizing negative log-likelihood on a held-out calibration set.
- **Platt scaling:** Apply a per-class affine transformation to logits before softmax: logit'_k = a_k · logit_k + b_k. More flexible than temperature scaling (which is a special case with a_k = 1/T, b_k = 0 for all k).
- **Brier score:** The mean squared error between predicted probability vectors and one-hot ground truth. Unlike ECE, it is a proper scoring rule (it's minimized by the true probabilities). Brier = Σ (p_k - y_k)² / n.
- **Exploration-exploitation tradeoff:** When the model is uncertain, it may be worth trying multiple modes. But exploration costs time/compute. The threshold for exploration depends on how calibrated the confidence is.

### Programming Prerequisites

- Python with NumPy and SciPy (especially `scipy.optimize.minimize_scalar` for temperature optimization)
- Matplotlib for reliability diagrams and threshold sweep plots
- JSON for data storage and management
- Basic PyTorch for loading the ModeSelector checkpoint and running inference

### Codebase Familiarity

- `ska_agent/router/adaptive_router.py` — `ModeSelector.predict()` returns (mode, probability_array). Understand that it applies softmax to logits internally.
- `ska_agent/core/structures.py` — `CollaborationMode` enum (LOOKUP, MULTI_DOC, COMPUTE, MULTI_STEP)

---

## Prep Phase (Weeks 1–3)

### Prep Week 1: Calibration Theory

**Goal:** Understand calibration deeply enough to explain it to a classmate and implement it from scratch.

**Study tasks:**
1. Read Guo et al. (2017), "On Calibration of Modern Neural Networks." This is the foundational paper on neural network calibration. Focus on: why modern networks are overconfident, how temperature scaling works, and how ECE is computed.
2. Implement ECE computation from scratch (not using a library). Test it on synthetic data: create fake probability predictions and labels where you control the calibration, and verify your ECE function gives the expected result.
3. Implement a reliability diagram plotting function. Test it: a perfectly calibrated model should produce points on the diagonal; an overconfident model should produce points below the diagonal.
4. Understand the difference between ECE, MCE (maximum calibration error — the worst bin), and Brier score. When might you prefer one over another?

**Deliverable:** Working implementations of ECE, MCE, Brier score, and reliability diagrams, tested on synthetic data with known calibration properties.

### Prep Week 2: Calibration Methods

**Goal:** Implement and understand all three post-hoc calibration methods.

**Study tasks:**
1. Implement temperature scaling: given a set of logits and true labels, find the temperature T that minimizes negative log-likelihood. Use `scipy.optimize.minimize_scalar` with bounds (0.1, 10.0).
2. Implement Platt scaling: learn a_k and b_k for each class using logistic regression on the logits. This requires a small optimization loop.
3. Implement histogram binning: bin predictions by confidence, then replace each bin's predicted probability with the bin's observed accuracy.
4. Test all three methods on synthetic calibration data. Create an overconfident model (logits are too large) and show that each method reduces ECE.

**Deliverable:** Working implementations of temperature scaling, Platt scaling, and histogram binning. A comparison table showing ECE before and after each method on synthetic data.

### Prep Week 3: Calibration Dataset and Experimental Plan

**Goal:** Build the evaluation infrastructure before the trained model arrives.

**Study tasks:**
1. Generate the calibration dataset: 400+ queries with ground-truth mode labels, using the template-based approach from the starter code. Split into calibration set (200+) and test set (100+).
2. Verify the dataset is balanced across 4 modes. Print 5 example queries per mode to check quality.
3. Plan the exploration policy: when the calibrated confidence is below a threshold, try the top 2 modes. Define how you'll measure the value of exploration (did trying the second mode produce a better result?).
4. Run the untrained ModeSelector on the calibration set to establish a "no training" baseline. With random weights, predictions should be roughly uniform across modes.

**Deliverable:** A saved calibration dataset (JSON). Baseline ECE on untrained model. Written plan for the exploration policy experiment.

---

## Build Phase (Weeks 4–10)

### Build Week 1 (Week 4): Baseline Calibration Analysis

**Tasks:**
- Load the S5-trained ModeSelector checkpoint.
- Run it on the full calibration set. Record logits (pre-softmax) and probabilities (post-softmax) for all queries.
- Compute ECE, MCE, and Brier score on the test set (not the calibration set — that's for fitting the calibration methods).
- Plot the reliability diagram. Determine whether the model is overconfident or underconfident.

**Expected output:** Pre-calibration ECE, MCE, and Brier score reported. Reliability diagram showing the calibration pattern. Most neural networks are overconfident, so you'll likely see points below the diagonal.

### Build Week 2 (Week 5): Apply Calibration Methods

**Tasks:**
- Apply temperature scaling: find optimal T on the calibration set, then measure ECE on the test set.
- Apply Platt scaling: fit per-class parameters on the calibration set, measure on test set.
- Apply histogram binning: fit bins on the calibration set, measure on test set.
- Produce a comparison table with 4 rows (uncalibrated, temperature, Platt, histogram) and columns for ECE, MCE, Brier score.
- Plot reliability diagrams for each method.

**Expected output:** Comparison table showing calibration improvement. At least 30% reduction in ECE from the best method. Reliability diagrams that are closer to the diagonal after calibration.

### Build Week 3 (Week 6): Calibrated Exploration Policy

**Tasks:**
- Implement the exploration policy: when calibrated confidence < threshold, run the top 2 modes and pick the better result.
- Define how to measure "better result." Options: use mock specialists and measure the number of actions taken, or use a quality proxy based on how well the retrieved context matches the query.
- Run 100 test queries through calibrated routing. Record: how often exploration fires, how often it discovers a mode that outperforms the top-1 prediction.

**Expected output:** Exploration policy working end-to-end. Metrics showing exploration rate and discovery rate.

### Build Week 4 (Week 7): Threshold Sweep

**Tasks:**
- Sweep the exploration threshold from 0.3 to 0.9 in steps of 0.1.
- At each threshold, measure: exploration rate (fraction of queries where the second mode is tried), mode accuracy (fraction of queries where the final mode is correct), and total cost (proportional to how many modes are tried).
- Plot threshold vs accuracy and threshold vs cost tradeoff curves.
- Identify the optimal threshold as the point where accuracy stops improving but cost keeps increasing (the "knee" of the tradeoff curve).

**Expected output:** Tradeoff curves plotted. Optimal threshold identified and justified.

### Build Week 5 (Week 8): Detailed Query Pattern Analysis

**Tasks:**
- Categorize queries by type and by whether exploration helped. What kinds of queries are the model most uncertain about? Are there systematic confusion patterns (e.g., LOOKUP vs MULTI_DOC confusion)?
- Compute per-mode calibration (ECE per mode). Some modes may be better calibrated than others.
- Analyze the relationship between calibrated confidence and actual correctness. Is there a confidence threshold below which the model is essentially random?

**Expected output:** Per-mode calibration metrics. Confusion patterns identified. A "confidence floor" below which the model's predictions are unreliable.

### Build Week 6 (Week 9): Exploration Value Analysis

**Tasks:**
- For queries where exploration fires, characterize them: are they genuinely ambiguous queries, or is the model just poorly trained on certain patterns?
- Compute the expected value of exploration: on average, how much does the second-mode try improve quality? Is this improvement worth the cost?
- Compare calibrated exploration vs two baselines: (1) always explore (try top 2 for every query) and (2) never explore (always trust top 1).

**Expected output:** Quantified value of calibrated exploration. Clear comparison against always-explore and never-explore baselines.

### Build Week 7 (Week 10): Written Analysis

**Tasks:**
- Write a 2–3 page analysis covering: calibration measurement, calibration methods comparison, exploration policy design, threshold optimization, and query pattern analysis.
- Include: reliability diagrams before and after calibration, threshold tradeoff curves, and specific examples of queries where exploration helped and where it didn't.
- Recommend: which calibration method to use, what exploration threshold to set, and when calibration-driven exploration is worth the cost.

**Expected output:** Complete written analysis with specific, actionable recommendations.

---

## What "Done" Looks Like

1. ECE/MCE/Brier score computed for the trained ModeSelector (pre- and post-calibration)
2. Three calibration methods implemented and compared, with the best one identified
3. A calibrated exploration policy that uses confidence to decide when to try alternative modes
4. Threshold sweep with tradeoff curves and an identified optimal threshold
5. Written analysis identifying query patterns where exploration helps and where it doesn't, with a recommended threshold and calibration method

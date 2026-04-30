# M2: Neural Sparsity Parameter Prediction — Full Project Guide

## Project Classification: Math/ML-Hybrid

**Tier:** Intermediate | **GPU:** Minimal | **Team Size:** 1 | **Total Duration:** 10 weeks (3 prep + 7 build)

**Dependencies:** None

---

## Project Summary

You are building a small MLP that predicts the optimal sparsity parameter (lambda) for the PricingEngine on a per-query basis. Currently lambda is fixed for all queries, but the optimal value depends heavily on query complexity — a simple factual lookup needs aggressive sparsity (high lambda, few segments), while a multi-hop comparison needs thorough retrieval (low lambda, many segments). You generate training data by sweeping lambda values, train the predictor, evaluate against fixed-lambda baselines, and integrate it into the retrieval pipeline.

---

## Motivation: Why This Matters for SKA-Agent

SKA-Agent's retrieval pipeline doesn't work like standard RAG. Instead of dumping the top-k most similar chunks into a context window, the PricingEngine uses a principled optimization framework: it adds segments one at a time, only including a segment if its information gain (measured via the Schur complement of the query-segment projection) exceeds a cost threshold set by lambda. This is what prevents the system from wasting context window space on redundant or marginally relevant segments — a critical property when the retrieved context is being compressed into a fixed-size Koopman operator for the shared spectral memory.

The problem is that lambda is currently a fixed scalar, but the optimal sparsity level varies dramatically by query. A simple lookup ("What was total federal debt in 2023?") needs only 1–2 highly targeted segments and benefits from high lambda. A multi-hop comparison ("Compare defense spending trends across three fiscal years and explain the drivers") needs broad coverage across many segments and requires low lambda. A fixed lambda is always wrong for some fraction of queries — either retrieving too much (wasting budget and diluting the operator) or too little (missing critical context).

This matters more for SKA-Agent than for a vanilla RAG system because the retrieved segments feed into the Koopman operator construction. Too many noisy segments degrade the operator's condition number; too few leave gaps that the power spectral filter can't compensate for. By learning lambda per query, you're not just optimizing retrieval — you're optimizing the quality of the spectral representation that the entire multi-agent system depends on.

---

## Starting Requirements

### Mathematical Prerequisites

- **Optimization basics:** Understand what a loss function is, what gradient descent does, and what "amortized optimization" means — instead of solving an optimization problem per query at inference time, you train a network to predict the solution directly.
- **The sparsity-quality tradeoff:** The PricingEngine's reduced cost formula is c̄(j) = λ + η·redundancy - information_gain(j). A segment is included if c̄(j) < 0. Higher λ means fewer segments pass the threshold. You need to understand this formula intuitively: lambda is a "price floor" for including a segment.
- **Log-space prediction:** Lambda spans several orders of magnitude (0.001 to 0.5). Training an MLP to predict lambda directly is hard because the loss is dominated by large values. Predicting log(lambda) and using MSE in log-space treats a 2× error at lambda=0.01 the same as a 2× error at lambda=0.1.
- **Evaluation metrics:** Precision (fraction of retrieved segments that are relevant), efficiency (fewer segments for the same quality), and the concept of an oracle baseline (the best lambda for each query, found by exhaustive search).

### Programming Prerequisites

- Python with NumPy for data generation and analysis
- PyTorch basics: defining an `nn.Module`, writing a training loop, train/val splits, saving/loading checkpoints
- JSON for data storage, Matplotlib for plotting
- Basic understanding of embeddings (sentence-transformers produces a 384-dim vector per query)

### Codebase Familiarity

- `ska_agent/core/pricing.py` — `PricingEngine.retrieve()` and `compute_reduced_cost()`. Understand the retrieval loop and where lambda appears.
- `ska_agent/core/structures.py` — `Segment` objects and their fields
- `ska_agent/models/embedding.py` — `Embedder` for converting text to vectors

---

## Prep Phase (Weeks 1–3)

### Prep Week 1: Understanding the PricingEngine

**Goal:** Build intuition for how lambda controls retrieval behavior.

**Study tasks:**
1. Read through `PricingEngine.retrieve()` step by step. Trace what happens when lambda=0.001 (many segments selected) vs lambda=0.5 (very few selected). Write down the decision boundary: a segment is included iff information_gain(j) > λ + η·redundancy.
2. Run the PricingEngine manually on a small example: create 10 synthetic segments, embed a query, and call `retrieve()` at lambda values [0.001, 0.01, 0.05, 0.1, 0.5]. Record how many segments are retrieved at each lambda.
3. Plot the number of retrieved segments vs lambda. This should be a monotonically decreasing step function. Understand where the "interesting" range is — the lambda values where the segment count actually changes.
4. Understand the concept of "optimal lambda": for a given query, the optimal lambda maximizes some quality metric while minimizing the number of retrieved segments. This is a tradeoff, and the "best" lambda depends on how you weight quality vs efficiency.

**Deliverable:** A notebook showing the segment-count-vs-lambda curve for 5 different queries. A written explanation of why different queries have different optimal lambda values.

### Prep Week 2: Embeddings and the Prediction Problem

**Goal:** Understand what information the MLP has access to and what it needs to predict.

**Study tasks:**
1. Use the sentence-transformers encoder to embed 20 diverse queries. Examine the embedding space: compute pairwise cosine similarities. Do similar queries (e.g., two LOOKUP queries) cluster together? Do queries of different complexity live in different regions?
2. Understand the MLP architecture: input is a 384-dim query embedding, output is a single positive number (lambda). The Softplus activation ensures lambda > 0. Think about what features of the embedding the MLP might learn to use (e.g., query length correlates with complexity, which correlates with optimal lambda).
3. Understand why MSE in log-space is the right loss. If the target lambda is 0.01 and you predict 0.02, that's a 2× error. If the target is 0.1 and you predict 0.2, that's also a 2× error. In linear space, MSE would treat the second error as 100× more important. Log-space treats them equally.
4. Plan the training data pipeline: for each query, you sweep lambda, measure a quality signal at each lambda, and pick the lambda that maximizes quality-efficiency. This becomes the training target.

**Deliverable:** A notebook showing query embeddings colored by type (LOOKUP, COMPUTE, etc.), and a written description of the prediction problem.

### Prep Week 3: Experimental Design

**Goal:** Plan the full experiment before writing production code.

**Study tasks:**
1. Design your quality signal. Since you don't have ground-truth answers for most queries, you need a proxy. Options: total reduced cost (sum of c̄ for selected segments — more negative is better), information coverage (how much of the query embedding is explained by the selected segments), or number of segments (as a proxy for cost, penalized by some quality floor).
2. Decide on your evaluation protocol: how will you compare learned lambda vs fixed lambda? You need a test set of queries (held out from training), a way to measure both quality and efficiency, and clear baselines (fixed lambda = 0.05, best single lambda from grid search, oracle per-query lambda).
3. Plan the cross-document robustness test: train on document A, test on document B. This matters because lambda depends on the segment distribution, which changes per document.
4. Write skeleton code for the full pipeline: data generation, training, evaluation, analysis.

**Deliverable:** A 1-page experimental plan specifying quality signal, baselines, evaluation metrics, and expected outcomes.

---

## Build Phase (Weeks 4–10)

### Build Week 1 (Week 4): Training Data Generation

**Tasks:**
- Generate 100 diverse queries using templates for all 4 mode types (LOOKUP, MULTI_DOC, COMPUTE, MULTI_STEP).
- For each query, sweep lambda across 8 values [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5].
- Record: number of segments retrieved, total reduced cost, and your chosen quality signal at each lambda.
- Identify the "optimal" lambda for each query (the one maximizing your quality-efficiency tradeoff).

**Expected output:** A JSON file with 800 records (100 queries × 8 lambdas). A histogram of optimal lambda per query showing meaningful variation (not all queries want the same lambda).

### Build Week 2 (Week 5): Train the LambdaPredictor

**Tasks:**
- Embed all queries using the sentence-transformers encoder.
- Split into 80% train / 20% validation.
- Train the LambdaPredictor MLP using MSE on log(lambda). Train for 200 epochs with Adam optimizer, lr=1e-3.
- Report validation MSE. Produce a scatter plot of predicted vs actual lambda.

**Expected output:** A trained model checkpoint. Validation MSE reported. The scatter plot should show positive correlation (the model is learning something), though it may not be perfect yet.

### Build Week 3 (Week 6): Expand Data and Retrain

**Tasks:**
- Expand to 300+ queries using more diverse templates and paraphrases.
- Retrain the MLP on the larger dataset.
- Verify the predictor is actually varying lambda by query — plot the distribution of predicted lambdas. If it's collapsed to a single value, the model hasn't learned anything useful.

**Expected output:** Improved validation MSE. A histogram of predicted lambdas that shows meaningful spread (not a spike at one value).

### Build Week 4 (Week 7): End-to-End Retrieval Comparison

**Tasks:**
- Define 4 retrieval configurations: (1) fixed lambda=0.05, (2) fixed lambda=best from grid search, (3) learned lambda from your MLP, (4) oracle lambda (best per query from sweep).
- Run all 4 on 100 held-out test queries.
- Measure: mean segments retrieved, mean total reduced cost, and retrieval precision (manually spot-check 20 queries for relevance).

**Expected output:** A comparison table showing that learned lambda matches or beats the best fixed lambda on at least one metric (precision or efficiency). The oracle sets an upper bound.

### Build Week 5 (Week 8): Lambda Analysis — What Did the Model Learn?

**Tasks:**
- Analyze the predictor's behavior: plot predicted lambda vs query word count, query type, and query embedding norm.
- Identify patterns: do short factual queries get high lambda (fewer segments)? Do complex multi-hop queries get low lambda (more segments)?
- Compute feature importance by perturbing input dimensions and measuring the effect on predicted lambda.

**Expected output:** Clear evidence that the MLP has learned a meaningful mapping from query complexity to sparsity level. At least one strong correlation (e.g., query length vs predicted lambda) documented.

### Build Week 6 (Week 9): Cross-Document Robustness

**Tasks:**
- Train on queries related to document A. Test on queries related to document B (different topic/structure).
- Measure performance degradation: how much worse is the predictor on unseen documents?
- Fine-tune on 10 examples from document B and measure recovery.

**Expected output:** Cross-document degradation quantified (e.g., "MSE increases 3× on unseen documents"). Fine-tuning on 10 examples recovers at least 50% of the performance gap.

### Build Week 7 (Week 10): Integration and Report

**Tasks:**
- Integrate the LambdaPredictor into PricingEngine by adding a `lambda_fn` parameter so the engine can accept a callable instead of a fixed value.
- Write usage documentation showing how to use `PricingEngine(lambda_fn=predictor)`.
- Write a 2–3 page report covering: the lambda prediction problem, training methodology, comparison results, analysis of what the model learned, cross-document robustness, and recommended usage.

**Expected output:** Working integration into PricingEngine. Written report with all key findings.

---

## What "Done" Looks Like

1. A trained LambdaPredictor MLP that varies lambda by query
2. A 4-way comparison (fixed default, fixed best, learned, oracle) with clear metrics
3. Analysis showing what the predictor learned about query-lambda relationships
4. Cross-document robustness evaluation with fine-tuning recovery results
5. Clean integration into PricingEngine with documentation
6. A written report summarizing all findings and recommending when learned lambda is beneficial vs when a fixed lambda suffices

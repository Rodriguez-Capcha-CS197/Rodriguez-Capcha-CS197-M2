# S5: Router Training Pipeline — Full Project Guide

## Project Classification: ML Engineering/Coding-Heavy

**Tier:** Intermediate | **GPU:** None (CPU training) | **Team Size:** 2 | **Total Duration:** 10 weeks (3 prep + 7 build)

**Dependencies:** None. THIS PROJECT IS THE DEPENDENCY for A1, A4, S4, and M4. You must deliver v1 checkpoints by the end of build week 3 (week 6 overall).

---

## Project Summary

You are building the data generation and training pipeline for the router's two learned components: the ModeSelector (4-way query classifier) and the RewardPredictor (marginal quality estimator). Four other teams depend on your trained checkpoints to make the router functional, so you have a hard deadline at build week 3.

**Person A** trains the ModeSelector. **Person B** trains the RewardPredictor. Both contribute to the shared `load_trained_router()` function and integration tests.

---

## Motivation: Why This Matters for SKA-Agent

The router is the decision-making brain of the entire system, and without trained components, it's brain-dead. The ModeSelector with random weights outputs roughly uniform probabilities across all four modes — it can't distinguish a simple lookup from a multi-step reasoning task. The RewardPredictor with random weights outputs near-zero scores for every specialist — so the scoring function S(a) = predicted_quality - λ^T · cost is always negative, and the router terminates immediately without invoking any specialist. The multi-agent system has all its pieces (retriever, code executor, reasoner, shared memory) but no way to activate them.

The router must make an up-front decision about which mode template to follow and which specialists to invoke, because the specialists are functionally different (a retriever can't do computation, a code executor can't do retrieval) and they communicate through the spectral memory in ways that depend on the execution order. The router's decisions determine the DAG structure of the entire multi-agent workflow — there's no fallback of "just try the next model in a list."

Your trained checkpoints are what transform the system from a collection of disconnected components into a functioning multi-agent pipeline. The ModeSelector learns to recognize query complexity from the embedding (short factual queries → LOOKUP, comparison queries → MULTI_DOC, calculation queries → COMPUTE, complex reasoning → MULTI_STEP), and the RewardPredictor learns which specialists are worth invoking for which query types. Four other project teams depend on these checkpoints — they literally cannot test their work without a functional router.

---

## Starting Requirements

### Technical Prerequisites (Coding-Focused)

- **PyTorch basics:** Defining nn.Module subclasses, writing training loops (forward pass, loss computation, backward pass, optimizer step), train/val splits, saving and loading state_dict checkpoints.
- **Sentence embeddings:** Using sentence-transformers to convert text queries into 384-dimensional vectors. Install with `pip install sentence-transformers`. The encoder is pretrained — you don't train it, you use it to create features for your MLPs.
- **Data generation:** Using templates and random filling to create diverse labeled datasets. Understanding why diversity matters for generalization.
- **Evaluation metrics:** Accuracy, confusion matrices, precision/recall per class (Person A). Pearson correlation, scatter plots of predicted vs actual (Person B).
- **Training diagnostics:** Recognizing underfitting (high training loss), overfitting (low training loss, high val loss), and learning rate issues (loss not decreasing, loss exploding).

### Tools

- Python, PyTorch, NumPy, Matplotlib
- sentence-transformers library
- JSON for data storage
- VIM or your preferred editor for code

### Codebase Familiarity

- `ska_agent/router/adaptive_router.py` — `ModeSelector` (2-layer MLP, 384→256→4) and `RewardPredictor` (3-layer MLP, 448→512→256→1). Read their `__init__` and `forward` methods.
- `ska_agent/training/trainers.py` — `RouterTrainer.train_mode_selector()` and `train_reward_predictor()`. These consume specific data formats.
- `ska_agent/core/structures.py` — `CollaborationMode` enum, `MODE_TEMPLATES`, `SystemConfig`

---

## Prep Phase (Weeks 1–3)

### Prep Week 1: Understanding the Router and Data Formats

**Goal:** Both persons understand the full router pipeline and their specific component's input/output contract.

**Person A tasks:**
1. Read `ModeSelector.__init__()` and `forward()`. The model takes a 384-dim query embedding, passes it through Linear(384,256) → ReLU → Linear(256,4), and returns softmax probabilities over 4 modes.
2. Understand the 4 modes: LOOKUP (simple factual queries), MULTI_DOC (cross-document comparison), COMPUTE (calculations needed), MULTI_STEP (complex multi-part reasoning). Write 10 example queries for each mode without looking at the templates.
3. Understand the training data format: `{"query_embedding": np.array(384,), "mode_idx": int}`. The trainer uses cross-entropy loss.
4. Set up the sentence-transformers encoder and verify it produces 384-dim embeddings.

**Person B tasks:**
1. Read `RewardPredictor.__init__()` and `forward()`. The model takes a concatenated vector (384-dim query embedding + 64-dim model embedding = 448 total), passes through Linear(448,512) → ReLU → Linear(512,256) → ReLU → Linear(256,1), and returns a scalar quality prediction.
2. Understand what the model predicts: delta_r = quality_of_specialist - quality_of_baseline. If delta_r > 0, the specialist is better than the baseline. The router uses this to decide which specialists to invoke.
3. Understand the training data format: `{"query_embedding": np.array(384,), "model_idx": int, "base_model_idx": int, "delta_r": float}`. The trainer uses MSE loss.
4. Understand the quality measurement setup: you need to measure how well each specialist handles each query, relative to a baseline (the parser).

**Deliverable:** Both persons can articulate their model's architecture, input format, loss function, and role in the router. Written in a shared document.

### Prep Week 2: VIM/CLI Basics and Template Design

**Goal:** Get comfortable with the development workflow and start designing training data.

**Shared tasks (both persons):**
1. **VIM essentials:** Open a Python file, navigate to a function, edit parameters, save and run. Practice: `vim train.py`, navigate to the learning rate, change it, `:wq`, `python train.py`. You need to be able to do this fluidly.
2. **CLI workflow:** Set up your project directory. Create a Makefile or shell scripts for: running training, running evaluation, generating data, running integration tests.

**Person A tasks:**
3. Design query templates for all 4 modes. Start with the provided templates, then add 5+ new templates per mode that cover different phrasing patterns.
4. Design the filler words: departments, years, financial terms, metrics. More diverse fillers = more diverse training data.
5. Generate 200 queries (50 per mode) and manually check 20 for quality. Are the mode labels correct? Are the queries natural-sounding?

**Person B tasks:**
3. Design the quality measurement methodology: for each query, how do you measure how well each specialist performs?
4. For the `ska_retriever`, quality = how well the retrieved segments match the query (projection-based scoring from starter code). For the `code_executor`, quality = whether the query needs computation (keyword-based proxy). For the `reasoner`, quality = whether the query is complex (length-based proxy). For the `parser`, quality is a low constant (it's the baseline).
5. Run the quality measurement on 20 queries manually. Verify the scores make intuitive sense.

**Deliverable:** Person A: 200 labeled queries saved to JSON. Person B: quality measurement function tested on 20 queries.

### Prep Week 3: First Training Runs

**Goal:** Complete a full training cycle before the build phase, so you can debug infrastructure issues early.

**Person A tasks:**
1. Embed all 200 queries using the sentence-transformers encoder.
2. Split 80/20 into train/val.
3. Train the ModeSelector for 200 epochs with Adam, lr=1e-3. Record training and validation loss.
4. Compute validation accuracy. It may be low with only 200 queries — that's okay. The goal is to verify the pipeline works end-to-end.

**Person B tasks:**
1. Create synthetic documents with known segments (use the OfflinePipeline on a small document).
2. Run quality measurements for 50 queries × 4 specialists = 200 data points.
3. Compute delta_r = quality_specialist - quality_parser for each.
4. Train the RewardPredictor for 100 epochs. Verify it produces non-zero predictions.

**Shared tasks:**
5. Implement `load_trained_router()` and verify it loads checkpoints correctly.
6. Run an integration test: load the trained router, call `route()` on 5 queries, verify it produces outputs (even if not great quality yet).

**Deliverable:** Both persons have a working training pipeline. Checkpoints saved. Integration test running.

---

## Build Phase (Weeks 4–10)

### Build Week 1 (Week 4): Expand Data

**Person A:**
- Expand to 500+ queries with paraphrasing and more template variety.
- Retrain. Measure accuracy improvement over the 200-query model.

**Person B:**
- Run retrieval pipeline on 100 queries with real segments from a processed document.
- Measure quality with the proxy scoring function for all 4 specialists.
- Train on expanded data.

**Expected output A:** 500+ queries, retrained model, accuracy reported.
**Expected output B:** 100-query quality measurements, retrained model.

### Build Week 2 (Week 5): Improve Quality

**Person A:**
- Error analysis on v1: which mode pairs get confused? (e.g., LOOKUP vs MULTI_DOC)
- Generate targeted examples for the confused pairs.
- Retrain. Target: 60%+ validation accuracy.

**Person B:**
- Ensure training data includes meaningful quality differences between specialists. The retriever should score much higher than the parser on retrieval-appropriate queries.
- Verify the predictor gives positive delta_r for good specialist-query pairs.

**Expected output A:** Confusion matrix, targeted examples generated.
**Expected output B:** Predictor produces positive scores for appropriate specialists.

### Build Week 3 (Week 6): CRITICAL — Ship v1 Checkpoints

**Person A deliverable:** `mode_selector_v1.pt` with at least 60% accuracy on a held-out test set. Non-uniform probability predictions (the model actually distinguishes between modes).

**Person B deliverable:** `reward_predictor_v1.pt` that produces positive scores for at least some actions, causing the router to take at least 1 action on 80% of test queries.

**Shared deliverable:** Announce to downstream teams (A1, A4, S4, M4). Share the `load_trained_router()` function and checkpoint files. Run integration test: router produces non-uniform mode probs and takes 1+ actions on 80% of test queries.

**This is a hard deadline. Four other teams are blocked without your checkpoints.**

### Build Week 4 (Week 7): Deeper Analysis

**Person A:**
- Full confusion matrix showing per-mode precision/recall.
- Identify the 3 most common error patterns and explain them.
- Expand to 750+ queries targeting weak spots.

**Person B:**
- Include quality measurements for all 4 specialists (not just retriever).
- Retrain on expanded data.
- Measure Pearson correlation between predicted and actual delta_r.

**Expected output A:** Confusion matrix, error patterns identified.
**Expected output B:** Correlation > 0.3 between predicted and actual quality.

### Build Week 5 (Week 8): Scale and Ablation

**Person A:**
- Expand to 1000+ queries. Retrain. Target: 75%+ accuracy.
- Ablation study: train on 100, 250, 500, 750, 1000 queries. Plot accuracy vs dataset size (learning curve).

**Person B:**
- Integration test: router with trained predictor makes multi-step decisions (2+ actions) on at least 50% of MULTI_STEP queries.
- Retrain on all available data. Measure correlation improvement.

**Expected output A:** Learning curve showing diminishing returns. Accuracy at 75%+.
**Expected output B:** Multi-step routing demonstrated. Correlation > 0.5.

### Build Week 6 (Week 9): Final Error Analysis

**Person A:**
- Examine 20 misclassified queries by hand. Categorize error types: ambiguous queries, misleading keywords, template artifacts.
- Suggest improvements for future training data.

**Person B:**
- Compare routing behavior with trained vs untrained predictor on 100 queries. Count: mode selection changes, action count differences, whether the trained router selects more appropriate specialists.

**Expected output A:** Error taxonomy with examples and improvement suggestions.
**Expected output B:** Comparison table showing trained router is meaningfully different from untrained.

### Build Week 7 (Week 10): Final Checkpoints and Report

**Person A:**
- Ship `mode_selector_v2.pt` (final, best checkpoint).
- Write methodology section: data generation approach, training details, accuracy results, error analysis, learning curve.

**Person B:**
- Ship `reward_predictor_v2.pt` (final, best checkpoint).
- Write methodology section: quality measurement approach, training details, correlation results, integration test results.

**Shared:**
- Verify all 4 downstream projects work with final checkpoints.
- Integration test passes with v2 checkpoints.

**Expected output:** Final checkpoints, written methodology, downstream verification.

---

## What "Done" Looks Like

### Person A
1. 1000+ labeled queries across 4 modes, with diversity beyond templates
2. ModeSelector with 75%+ validation accuracy
3. Confusion matrix and error taxonomy
4. Learning curve (accuracy vs dataset size) showing diminishing returns
5. v1 checkpoint delivered on time (week 6), v2 checkpoint at project end

### Person B
1. Quality measurements for 100+ queries across 4 specialists
2. RewardPredictor with Pearson correlation > 0.5 between predicted and actual quality
3. Router making multi-step decisions on complex queries
4. Comparison of trained vs untrained routing behavior
5. v1 checkpoint delivered on time (week 6), v2 checkpoint at project end

### Shared
1. `load_trained_router()` working and used by 4 downstream projects
2. Written methodology covering both components

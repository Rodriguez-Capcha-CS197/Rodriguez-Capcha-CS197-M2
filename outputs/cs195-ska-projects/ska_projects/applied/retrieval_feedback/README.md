# A4: Retrieval Quality Feedback Loop — Full Project Guide

## Project Classification: ML Engineering/Coding-Heavy

**Tier:** Intermediate | **GPU:** Minimal | **Team Size:** 1–2 | **Total Duration:** 10 weeks (3 prep + 7 build)

**Dependencies:** Requires trained router checkpoints from S5 Router Training (delivered at S5's build week 3). Build the feedback UI and storage in prep and early build weeks with mock data, then integrate the real router.

---

## Project Summary

You are building a human-in-the-loop feedback system for the router. Users rate retrieval quality (thumbs up/down on individual segments, star ratings on answers), you store that feedback in a database, convert it into training data for the RewardPredictor and ModeSelector, run a training cycle, and measure whether the feedback-trained router makes better decisions than the original. This is a lightweight version of RLHF applied to the routing layer.

---

## Motivation: Why This Matters for SKA-Agent

SKA-Agent's router makes two learned decisions on every query: which collaboration mode to use (ModeSelector) and which specialists are worth invoking (RewardPredictor). Both models are initially trained on synthetic data and proxy metrics — the ModeSelector on template-generated queries with known labels, the RewardPredictor on automated quality measurements. This bootstrapping gets the system functional, but it's limited by the quality of the proxies. The real question is whether the retriever actually helped the user, whether the code executor produced a correct calculation, and whether the selected mode led to a good answer. Only human judgment can provide that signal.

This is where SKA-Agent's architecture creates a unique opportunity. Because the system routes to functionally distinct specialists through structured DAG templates, you can get granular feedback at multiple levels: thumbs up/down on individual retrieved segments (was this segment relevant?), star ratings on the final answer (was the overall workflow successful?), and implicit signals from mode selection (the router chose LOOKUP but the user's follow-up question suggests it should have been MULTI_STEP). This multi-level feedback maps directly onto the router's learned components.

The feedback loop you build closes a critical gap in the system. The S5 team trains the initial router on synthetic data. Your project measures how real users experience the router's decisions, converts that experience into training signal, and fine-tunes the models. The lambda tuning component is particularly valuable: by analyzing which queries had too many irrelevant segments (lambda too low) or missing context (lambda too high), you produce per-mode lambda recommendations that the PricingEngine can use to adapt retrieval to query type — connecting the routing layer to the retrieval layer through human feedback.

---

## Starting Requirements

### Conceptual Prerequisites

- **Human feedback as training signal:** The RewardPredictor needs (query, specialist, quality) tuples to learn which specialists are good for which queries. Currently this data comes from proxy metrics. User feedback provides a more direct signal: "this segment was useful" (thumbs up) or "this was irrelevant" (thumbs down).
- **Converting ratings to training labels:** A 5-star answer rating maps to a quality delta: 5 stars = +1.0 (specialist was very helpful), 3 stars = 0.0 (neutral), 1 star = -1.0 (specialist was harmful). The formula delta_r = (rating - 3) / 2.0 normalizes to [-1, +1].
- **A/B comparison:** To measure improvement, you run the same queries through the original router and the feedback-trained router, then compare: how often they select different modes, how many actions each takes, whether the feedback-trained version avoids known failure patterns.
- **Lambda tuning from feedback:** If users consistently rate most retrieved segments as irrelevant (many thumbs down), lambda was too low (too many segments retrieved). If users say context was missing, lambda was too high. This gives a per-mode lambda recommendation.

### Technical Prerequisites (Coding-Focused)

- **Python and Streamlit:** For the feedback UI.
- **SQLite:** A simple file-based database. You need: `CREATE TABLE`, `INSERT`, `SELECT`, basic SQL queries. SQLite is built into Python (`import sqlite3`).
- **PyTorch basics:** Loading checkpoints, running a training loop, saving updated checkpoints. You're fine-tuning an existing model, not building one from scratch.
- **JSON:** For data export and training data format.
- **Data analysis:** Aggregating feedback by mode, computing average ratings, plotting distributions.

### Codebase Familiarity

- `ska_agent/router/adaptive_router.py` — `RewardPredictor` (what you're improving with feedback) and `ModeSelector` (also improvable from feedback).
- `ska_agent/training/trainers.py` — `RouterTrainer.train_reward_predictor()` and `train_mode_selector()`. You generate data in the format these functions expect.
- `ska_agent/core/pricing.py` — `PricingEngine`, where lambda affects retrieval. You'll recommend per-mode lambda values.

---

## Prep Phase (Weeks 1–3)

### Prep Week 1: Feedback UI Design

**Goal:** Build the UI components for collecting feedback.

**Tasks:**
1. Build a segment feedback widget: for each retrieved segment, display the text with thumbs up/down buttons. Use Streamlit columns for layout.
2. Build an answer rating widget: a 1–5 star slider with descriptions (1 = wrong, 3 = partial, 5 = perfect).
3. Build a basic query interface: text input → retrieve segments → display with feedback widgets → collect ratings.
4. Wire up session state to collect all feedback for a single query: segment-level ratings and answer-level rating.
5. Test the UI flow: type a query, see segments, rate them, rate the answer. All ratings captured in session state.

**Deliverable:** Working feedback UI with segment thumbs up/down and answer star rating. All ratings captured.

### Prep Week 2: VIM/CLI, SQLite Storage

**Goal:** Persist feedback in a database so it survives across sessions.

**Tasks:**
1. **VIM practice:** Edit the database schema, modify SQL queries, add new columns.
2. Set up SQLite: create the feedback table with columns for timestamp, query, query_embedding (as blob), mode_selected, lambda_used, num_segments, segment_ratings (as JSON), answer_rating, and answer_text.
3. Implement `save_feedback()`: after the user submits ratings, write all data to the database.
4. Implement a feedback review page: query the database and display all past feedback entries in a table.
5. Verify: submit 10 feedback entries, query them back, verify all fields are populated correctly.

**Deliverable:** SQLite database with save and retrieve functions. 10 test entries stored and retrievable.

### Prep Week 3: Feedback Dashboard and Analysis

**Goal:** Build the analytics view and prepare for training data generation.

**Tasks:**
1. Build a feedback dashboard: average rating over time (line chart), rating distribution (histogram), per-mode breakdown (bar chart by mode showing average rating).
2. Verify the dashboard updates as new feedback is added.
3. Implement the training data generation functions: `generate_reward_training_data()` and `generate_mode_training_data()`. These read from the database and produce the format expected by `RouterTrainer`.
4. Test the data generation on your 10 test entries. Verify the output format matches what the trainer expects.

**Deliverable:** Feedback dashboard with summary statistics. Training data generation functions producing correctly formatted data.

---

## Build Phase (Weeks 4–10)

### Build Week 1 (Week 4): Real Router Integration and Data Collection

**Tasks:**
- Load S5 checkpoints. Replace mock retrieval with real router decisions.
- Collect feedback on real routing results. Aim for 50+ feedback entries by end of week (self-annotate: run queries, judge the results, provide ratings).
- Tips for efficient annotation: prepare a list of 60 diverse queries beforehand, spend 1–2 minutes per query.

**Expected output:** 50+ entries in the database. Router producing different modes for different queries. Feedback capturing real quality variation.

### Build Week 2 (Week 5): Training Data Validation

**Tasks:**
- Run `generate_reward_training_data()` on the 50+ entries. Verify format: each entry has query_embedding (384-dim), model_idx, base_model_idx, and delta_r.
- Run `generate_mode_training_data()` on high-rated entries (4+ stars). Verify format: query_embedding and mode_idx.
- Analyze the training data: what's the distribution of delta_r values? Are there enough positive and negative examples? Is mode distribution balanced?

**Expected output:** Training data validated. Distribution analysis showing the data has signal (not all neutral ratings).

### Build Week 3 (Week 6): First Training Cycle

**Tasks:**
- Fine-tune the RewardPredictor starting from the S5 v1 checkpoint. Use the feedback-derived training data. Train for 100 epochs with a lower learning rate (5e-4 to avoid catastrophic forgetting).
- Compare predictions: run 20 test queries through both the original and fine-tuned predictor. Do the scores change? Are the changes in the right direction (higher scores for specialists that users rated highly)?

**Expected output:** Fine-tuned predictor checkpoint. Evidence that predictions changed in response to feedback.

### Build Week 4 (Week 7): A/B Comparison

**Tasks:**
- Build the A/B comparison: run 50 queries through both the original and feedback-trained routers.
- Measure: mode agreement rate (how often they pick the same mode), action count comparison, and whether the feedback-trained router avoids patterns that got low ratings.
- Display results in a comparison table and summary statistics.

**Expected output:** Comparison table with 50 rows. Summary showing agreement rate and key differences.

### Build Week 5 (Week 8): Lambda Tuning from Feedback

**Tasks:**
- Analyze segment-level feedback: for queries where most segments got thumbs down, lambda was probably too low (too many irrelevant segments retrieved). For queries where the answer was rated 1–2 stars with "missing context" comments, lambda was probably too high.
- Group queries by mode. For each mode, compute the average segment approval rate (fraction of segments rated thumbs up). Modes with low approval rates need higher lambda; modes with high approval rates can keep current lambda.
- Produce a per-mode lambda recommendation table.

**Expected output:** Per-mode lambda recommendations based on feedback patterns. E.g., "LOOKUP: lambda=0.1, COMPUTE: lambda=0.02."

### Build Week 6 (Week 9): Iterative Improvement

**Tasks:**
- Collect another round of feedback (20+ entries) using the feedback-trained router. Are ratings improving?
- Run a second training cycle on the accumulated feedback. Compare v1 (original) → v2 (first feedback cycle) → v3 (second feedback cycle).
- Analyze convergence: is the feedback loop improving, plateauing, or degrading?

**Expected output:** Multi-round comparison showing feedback loop trajectory. Evidence of improvement or plateau.

### Build Week 7 (Week 10): Written Analysis

**Tasks:**
- Write a 2–3 page analysis covering: the feedback collection process, training data generation methodology, A/B comparison results, lambda tuning recommendations, and the multi-round feedback loop.
- Include: rating distribution charts, comparison tables, lambda recommendations per mode, and an honest assessment of whether the feedback loop improved routing.
- Discuss limitations: how much feedback is needed to see improvement? How does annotation quality affect results?

**Expected output:** Written report with honest findings about the value of the feedback loop.

---

## What "Done" Looks Like

1. Feedback UI with segment-level thumbs up/down and answer-level star rating
2. SQLite storage capturing full metadata (query, embedding, mode, lambda, ratings)
3. Training data generation producing correctly formatted data for RouterTrainer
4. At least one complete training cycle (feedback → training data → fine-tune → evaluate)
5. A/B comparison quantifying the effect of feedback-trained routing
6. Per-mode lambda recommendations derived from segment-level feedback
7. Written analysis documenting the full feedback loop and its impact

# CS195: Spectral Koopman Attention — Student Projects

## What This Course Project Track Is About

This repository contains learning materials and 13 independent research projects exploring **Spectral Koopman Attention (SKA)**, a method that replaces standard transformer attention with Koopman operators — small, fixed-size matrices that compress an arbitrarily long input sequence into a spectral memory that doesn't grow with context length.

Standard attention stores a key-value pair for every token. For a sequence of length T with head dimension d, that's O(T·d) memory per layer, and the attention computation itself is O(T²). This is why transformers struggle with long contexts: at 256K tokens, the KV cache alone can exceed 30GB. SKA replaces this with an r×r operator (typically r = 64) that captures the same temporal dynamics in O(r²) memory — **independent of context length**. Whether the input has 100 tokens or 262,000 tokens, the operator is the same size.

---

## The Core Idea: Koopman Operators for Attention

### What Is a Koopman Operator?

The Koopman operator comes from dynamical systems theory. Given a nonlinear dynamical system where states evolve over time, the Koopman operator is an infinite-dimensional linear operator that exactly represents the dynamics — but in a lifted function space rather than the original state space. The key insight is that **nonlinear dynamics become linear** when viewed through the right lens.

For sequence modeling, we treat consecutive token representations as a discrete dynamical system: token t evolves into token t+1 through some (nonlinear) function of the transformer's hidden states. The Koopman operator approximation captures this evolution as a matrix multiply — and because it's linear, we can analyze its spectral properties (eigenvalues, singular values) to understand what the sequence is doing.

### How SKA Uses This

In standard multi-head attention, given a prefix of L tokens, each query attends to all L keys and retrieves a weighted combination of values. SKA replaces this with a three-step pipeline:

**Step 1 — Build the operator from prefix keys:**

Given prefix key vectors z₁, z₂, ..., z_L in R^r (projected from the hidden states):

- **Gram matrix:** G = Σ zₜ zₜᵀ + εI — a regularized summary of the key distribution. The ridge term ε ensures G is always invertible.
- **Transition matrix:** M = Σ z_{t+1} zₜᵀ — captures how keys evolve from one position to the next. This is where temporal structure enters: M encodes the sequential dynamics of the prefix.
- **Cholesky factorization:** G = LLᵀ — the Cholesky factor L is the "square root" of G. It defines a whitening transform that normalizes the key space.
- **Whitened Koopman operator:** A_w = L⁻¹ M L⁻ᵀ — computed via triangular solves, never forming explicit inverses. This is a similarity transform of the natural operator MG⁻¹, meaning the eigenvalues are preserved but the computation is numerically stable.

**Step 2 — Spectral normalization:**

The operator is scaled so its largest singular value doesn't exceed γ ≤ 1. This is critical because step 3 applies the operator repeatedly (power iteration), and if σ_max > 1, the output would explode.

**Step 3 — Power spectral filtering of queries:**

For each query z_q:
1. **Whiten:** w_q = L⁻¹ z_q (triangular solve)
2. **Power filter:** w_f = A_w^K w_q (apply the operator K times, default K=2)
3. **Unwhiten:** ẑ = L w_f (matrix multiply)
4. **Read out values:** ŷ = B_v ẑ (where B_v = C_v G⁻¹ is the value readout, also computed via triangular solves)

The power filter is the key mechanism: directions in key space that were **consistent across the prefix** (high eigenvalues of A_w) are amplified, while directions that fluctuated randomly (low eigenvalues) are suppressed. This is spectral filtering — the operator separates signal from noise in the temporal structure of the prefix, and the query reads through this filter.

### Why This Is Useful

The operator A_w is r×r regardless of how long the prefix was. A 64×64 matrix occupies ~32KB. Compare this to a standard KV cache for a 256K-token prefix at head dimension 128: that's 256K × 128 × 2 (keys + values) × 4 bytes ≈ 256MB per layer. SKA compresses this to 32KB — a **8,000× reduction** — while preserving the temporal dynamics that attention would have captured.

This fixed-size property is what enables several of the projects in this repository: incremental operator updates for streaming memory (S1), weight transplantation from pretrained models (M1), streaming segmentation with bounded memory (S2), and retrieval-augmented generation through the pricing engine (A3).

---

## The Koopman MLP: A Simpler Variant

Not every project uses the full SKA attention replacement. Several projects work with the **Koopman MLP**, a simpler module that applies the same Koopman operator idea to feedforward layers:

1. Project a hidden state into a low-rank space
2. Apply a learned Koopman matrix (with spectral normalization)
3. Project back to the original dimension

This lets you study the Koopman operator's properties (spectral structure, initialization, regularization) without the complexity of the full attention replacement. The math-track projects (M1–M4) investigate these properties in depth.

---

## SKA-Agent: The Full System Built on SKA

The projects in this repository study SKA as a standalone attention mechanism, but SKA was designed as the foundation for a larger system: **SKA-Agent**, an adaptive multi-model orchestration framework where multiple specialist models collaborate on complex queries, communicating through shared Koopman operators instead of passing text back and forth.

### The Problem SKA-Agent Solves

Consider a question like: *"Compare defense spending as a percentage of GDP across the last three fiscal years, calculate the average annual growth rate, and explain the primary drivers."* This requires three different capabilities: document retrieval (finding the right numbers in budget reports), computation (calculating growth rates), and reasoning (explaining the drivers). No single model is optimal for all three — a small retrieval specialist is fast and cheap for finding facts, a code executor handles calculations exactly, and a large reasoning model synthesizes the explanation.

The naive approach is to dump everything into one model's context window: paste the retrieved documents, the calculation results, and the original question into GPT-4 and hope for the best. This works, but it's expensive (the context window fills up fast), brittle (irrelevant retrieved passages dilute the signal), and opaque (you can't see which component contributed what).

SKA-Agent takes a different approach. It routes queries to functionally distinct specialists — a retriever, a code executor, a reasoner — and the specialists share what they learn through **spectral memory**: fixed-size Koopman operators that compress each specialist's findings into a small matrix that other specialists can query.

### The Three-Tier Architecture

SKA-Agent uses three tiers of models, each with different cost and capability profiles:

**Tier 1 — Qwen Coordinator (27B, quantized to ~16.5GB):** The primary reasoner and coordinator. It handles task decomposition (breaking a complex goal into subtasks), structured reasoning (with `<think>` chain-of-thought traces), and final answer synthesis. Its reasoning traces are projected into the shared spectral memory via the Think-Koopman Bridge, so other agents can query what the coordinator is thinking about without reading the full text of its reasoning chain.

**Tier 2 — Jamba+SKA (12B active, retrieval specialist):** A Jamba model where 4 attention layers (at positions 4, 12, 20, 28) have been surgically replaced with SKA modules. This is the retrieval specialist: it ingests documents through the SKA operator and produces dense, compressed representations that the pricing engine can query. The SKA surgery process extracts the pretrained attention weights, runs truncated SVD to find the top-r directions, and initializes the SKA projections with sqrt-singular-value scaling — preserving the model's learned representations while fitting them into the fixed-size operator framework.

**Tier 3 — DeepSeek V3 (optional, for hard multi-hop reasoning):** A heavy reasoning model invoked only for queries that exceed the coordinator's capabilities. Cost-gated by the PID controller so it's only called when the expected quality improvement justifies the expense.

### How Specialists Communicate: Shared Spectral Memory

This is the architectural innovation that makes SKA-Agent more than just "a router that picks which model to call." In a conventional multi-agent system, Agent A writes 2,000 tokens of findings into Agent B's context window — growing B's KV cache by ~32MB per handoff and consuming precious context space. In SKA-Agent, Agent A writes key vectors that update a rank-r Koopman operator (~32KB), and Agent B queries that operator through the power spectral filter to retrieve what A learned.

The shared memory has multiple slots organized as a multi-head Koopman module (K=4 parallel rank-48 operators per attention head):

- **Slots 0–1:** Document structure (written by the parser specialist)
- **Slot 2:** Reasoning state (written by the coordinator via the Think-Koopman Bridge, which projects the coordinator's `<think>` hidden states into operator space)
- **Slot 3:** Temporal patterns

Each slot has its own key projection, Gram matrix, and gate. Slot specialization emerges during training — different agents naturally write to different slots because their key distributions are different, and the per-slot gates learn to weight the slots appropriately for each query type.

The operator captures temporal dynamics: the transition matrix M = Σ z_{t+1} zₜᵀ encodes how an agent's findings evolved across steps. Directions where reasoning was consistent (high eigenvalues) are amplified by the power filter; directions that fluctuated randomly (low eigenvalues) are suppressed. So the operator doesn't just store *what* an agent concluded — it stores *how confident* that conclusion is, encoded in the spectral structure.

### The Router: Sequential Marginal Evaluation

The router decides which specialists to invoke and in what order. It has four learned components:

**QueryEncoder** (frozen MiniLM-L6-v2, 22M params): Encodes the input query into a 384-dim dense vector. Not trained — it's a pretrained sentence transformer used as a fixed feature extractor.

**ModeSelector** (2-layer MLP, ~100K params): Classifies the query into one of four collaboration modes — LOOKUP (simple factual question), MULTI_DOC (cross-document comparison), COMPUTE (calculation needed), MULTI_STEP (complex multi-part reasoning). Each mode defines a DAG template of valid transitions between specialist types.

**RewardPredictor** (3-layer MLP, ~400K params): Predicts the quality improvement from choosing specialist m over a baseline: Δr̂(a) = g(e_Q, e_m − e_base). This is the marginal value estimate: "how much better will the answer be if we invoke the retriever vs. just using the parser?"

**PID Controller** (no learned params): Adjusts a 5-dimensional price vector λ = (λ_in, λ_out, λ_lat, λ_$, λ_meta) based on the running cost rate. The five dimensions correspond to input tokens, output tokens, latency, dollar cost, and shared memory overhead.

The decision rule at each step: enumerate candidate actions from the current DAG node, score each as S(a) = Δr̂(a) − λᵀ · Δĉ(a), and execute the highest-scoring action if its score is positive. If no action has positive score, stop. The execution graph isn't predicted up front — it emerges from a sequence of local positive-score decisions. Simple queries terminate after 1–2 actions; complex multi-hop queries chain 4–5 specialists.

### The Retrieval Pipeline: Geometry Learning + Pricing-Guided Selection

Before any model inference happens, documents are preprocessed through a two-stage pipeline:

**Stage I — Geometry Learning:** Raw text is split into sentences, embedded, and segmented using dynamic programming. Unlike fixed-size chunking (every 512 tokens), the DP finds natural topic boundaries by minimizing internal pairwise cosine distance within each segment, subject to a sparsity penalty λ per segment. The result is semantically coherent atomic units that respect the document's natural structure. The DP runs in O(N·K) time where N is the sentence count and K is the lookback window (default 50).

**Stage II — Pricing-Guided Retrieval:** Given a query, the PricingEngine selects segments using a greedy optimization: the reduced cost for adding segment j is c̄(j) = λ + η·redundancy − information_gain(j), where the information gain is the Schur complement of the query-segment projection. A segment is included only if c̄(j) < 0 — meaning its information gain exceeds the cost threshold. After each selection, the query residual is updated by orthogonal projection (removing the explained component), which automatically prevents redundancy: once a segment's information is "used up," subsequent segments covering the same topic have near-zero information gain.

### Why This Matters for the Projects

Every project in this repository explores a piece of this larger system:

- **S1 and M1** directly implement and study core SKA-Agent components (incremental Cholesky updates for streaming memory, initialization strategies for the Jamba surgery process).
- **M2, M3, and M4** study the mathematical properties (sparsity prediction, regularization landscape, calibration) that determine whether the routing and retrieval systems work reliably.
- **S2, S3, S4, and S5** build or optimize infrastructure (streaming segmentation, TypeScript orchestrator, PID tuning, router training) that the full system depends on.
- **A1, A2, A3, and A4** build user-facing tools (dashboards, demo apps, feedback loops) that make the system visible, testable, and improvable.

Understanding SKA-Agent gives you context for *why* your project matters: you're not studying a matrix factorization in isolation — you're studying a component of a system where multiple models collaborate through spectral memory, and the reliability of that collaboration depends on the mathematical properties you're investigating.

---

## The Math You Need to Know

The first two weeks of the course build up the mathematical foundations. Here's what you'll learn and why it matters:

**Singular Value Decomposition (SVD):** Every matrix A can be written as A = UΣVᵀ where U and V are orthogonal and Σ is diagonal with non-negative entries (the singular values). The singular values tell you the "strength" of the matrix in different directions. In SKA, SVD is used for: (a) initializing key/query projections from pretrained attention weights (M1), (b) spectral normalization of the operator, and (c) analyzing what the operator has learned (M3).

**Cholesky Factorization:** A symmetric positive definite matrix G can be uniquely factored as G = LLᵀ where L is lower triangular. This is the numerically stable way to "invert" the Gram matrix — instead of computing G⁻¹ explicitly (which amplifies errors), we solve triangular systems Lx = b, which is O(r²) and backward-stable. Every operator construction in SKA goes through Cholesky. S1 extends this to incremental rank-1 updates.

**Condition Number:** κ(G) = σ_max/σ_min measures how sensitive the solution of Gx = b is to perturbations in G. When κ is large, the Cholesky factorization amplifies numerical errors, and the whitened operator A_w becomes unreliable. The ridge regularization εI in the Gram matrix keeps κ bounded — this is the same idea as Tikhonov regularization in inverse problems.

**Power Iteration and Spectral Filtering:** Repeatedly multiplying a vector by a matrix A amplifies the components aligned with A's dominant eigenvectors and suppresses the rest. After K applications, the output is dominated by the top-K eigenspaces. In SKA, this is how the query "reads" the prefix: the power filter A_w^K amplifies temporally consistent patterns and suppresses noise.

---

## Setup

```bash
pip install torch transformers datasets matplotlib numpy scipy
pip install sentence-transformers streamlit plotly
pip install scikit-optimize rank_bm25
```

Additional packages for specific projects:

- **S3 (TypeScript Orchestrator):** Requires Node.js 18+ and TypeScript (`npm install typescript ts-node`)
- **A1, A2, A3, A4 (dashboard projects):** `pip install streamlit plotly` (included above)
- **M1 (SVD Init Ablation):** One-time GPU access needed for weight extraction

All projects reference the `ska_agent/` codebase. The project guide files explain which modules each project interacts with.

---

## Repository Structure

Projects are organized into three tracks based on what they focus on:

```
ska_agent/               The SKA-Agent codebase (shared library)
    core/                SKA module, geometry learner, pricing engine, structures
    models/              Jamba+SKA surgery, Qwen coordinator, embedder
    router/              Adaptive router, PID controller
    shared_memory/       Spectral memory protocol, Think-Koopman bridge
    training/            SKA trainer, router trainer
    orchestration/       TypeScript integration layer (ToolServer)
    pipeline.py          End-to-end pipeline wiring

guides/                  Detailed project guides (one per project)

--- Math/Theory Track (M) ---
M1_svd_init_ablation     Compare 7 initialization strategies for SKA surgery
M2_learned_lambda        Neural prediction of per-query sparsity parameter
M3_regularization        Spectral and orthogonal regularization interaction
M4_mode_calibration      ModeSelector calibration + exploration policy

--- Systems/Engineering Track (S) ---
S1_incremental_cholesky  O(r²) incremental Koopman operator updates
S2_streaming_segmentation  Streaming bounded-memory DP segmentation
S3_ts_orchestrator       TypeScript orchestrator with parallel DAG execution
S4_pid_autotuning        PID cost controller gain optimization
S5_router_training       Router training pipeline (CRITICAL PATH dependency)

--- Application Track (A) ---
A1_router_dashboard      Router decision visualization + failure analysis
A2_memory_inspector      Spectral memory debugger + anomaly detection
A3_officeqa_demo         End-to-end document QA + retrieval comparison
A4_retrieval_feedback    Human feedback loop for router improvement
```

---

## The 13 Projects

### Math/Theory Track

These projects investigate the mathematical properties of the Koopman operator, the regularization landscape, and the learned components of the routing system. Students on the math track spend their prep weeks building foundations in linear algebra, spectral theory, and optimization before running empirical studies.

**M1 — SVD Initialization Ablation for SKA Surgery** (Intermediate, ~1hr GPU)
The SKA surgery process replaces attention layers with Koopman operators, initializing the key projections from the original attention weights via SVD. But sqrt-Σ scaling is just one of many possible strategies. You implement 7 initialization strategies (random orthogonal, random Gaussian, SVD with three different scalings, PCA, NMF), analyze their spectral properties (Gram matrix condition numbers, spectral gaps), run small-scale training comparisons across multiple ranks, and produce a data-backed recommendation for the default strategy.

**M2 — Neural Sparsity Parameter Prediction** (Intermediate, minimal GPU)
The PricingEngine uses a fixed lambda to control the sparsity-quality tradeoff in retrieval. You train a small MLP that predicts the optimal lambda per query — high lambda for simple lookups (few segments), low lambda for complex multi-hop queries (thorough retrieval). You generate training data by sweeping lambda values, train and evaluate the predictor against fixed-lambda baselines and an oracle, analyze what the model learns about query complexity, and test cross-document robustness.

**M3 — Spectral and Orthogonal Regularization Interaction** (Advanced, ~2hrs GPU)
Two regularization losses govern operator health during training: spectral regularization (penalizing ||A_w||_F², which controls operator strength) and orthogonal regularization (penalizing ||W_K^T W_K − I||, which controls input conditioning). You run a 25-point grid sweep over their joint space, produce heatmaps of 5 health metrics, identify Pareto-optimal configurations with sensitivity analysis, characterize the behavior at the four extreme corners, and recommend default parameters.

**M4 — Mode Selector Calibration + Calibrated Routing** (Intermediate, minimal GPU)
Neural networks are notoriously overconfident. If the ModeSelector says "95% LOOKUP" but is only right 60% of the time, the router trusts it too much and never explores alternatives. You measure calibration (ECE, reliability diagrams), apply three post-hoc calibration methods (temperature scaling, Platt scaling, histogram binning), then build an exploration policy that uses calibrated confidence to decide when to try multiple modes.
*Depends on S5 checkpoints at week 6.*

### Systems/Engineering Track

These projects build or optimize core infrastructure components. Students on the systems track spend their prep weeks learning tools (VIM, TypeScript, NumPy internals) and algorithms (Cholesky updates, topological sort, PID control) before writing production code.

**S1 — Incremental Cholesky Operator Updates** (Advanced, no GPU)
Pure numerical linear algebra. The shared spectral memory rebuilds the Koopman operator from scratch (O(r³)) on every write. You implement O(r²)-per-step incremental updates using rank-1 Cholesky updates with Givens rotations, propagate those rotations to the cached whitened operator, track numerical stability over 100K+ updates, implement refactorization scheduling, and benchmark speedups at ranks 32–512.

**S2 — Streaming Bounded-Memory Segmentation** (Intermediate, minimal GPU)
The geometry learner's batch DP loads all sentences before segmenting. You build a streaming version that processes sentences one at a time with O(lookback_k) memory, produces identical segments to the batch algorithm, and emits finalized segments incrementally. This enables the offline pipeline to process arbitrarily large corpora without running out of memory.

**S3 — TypeScript Orchestrator Integration** (Advanced, no GPU)
The Python ToolServer has endpoints for reasoning, retrieval, and code execution, but no client. You build a TypeScript orchestrator that decomposes goals into task DAGs, uses topological sort (Kahn's algorithm) to identify parallel groups, dispatches tasks via HTTP with retries and timeouts, synchronizes shared memory, and handles partial failures gracefully. Team project (2–3 students).

**S4 — PID Controller Auto-Tuning** (Intermediate, minimal GPU)
The PID controller's gains (Kp, Ki, Kd) are hand-tuned. You build an automated system: grid search over the 3D gain space, Bayesian optimization for sample-efficient search, the Ziegler-Nichols heuristic from control theory, and budget scheduling policies (constant, front-loaded, cosine). You compare all methods and recommend optimal gains with a specific budget schedule.
*Depends on S5 checkpoints at week 6.*

**S5 — Router Training Pipeline** (Intermediate, no GPU) **CRITICAL**
Four other projects (A1, A4, S4, M4) depend on your trained checkpoints. You build the data generation and training pipeline for the ModeSelector (4-way query classifier) and RewardPredictor (marginal quality estimator). Person A trains the ModeSelector on 1000+ template-generated queries. Person B trains the RewardPredictor on specialist quality measurements. **v1 checkpoints must ship by end of week 6.** Team project (2 students).

### Application Track

These projects build user-facing tools and conduct empirical analyses. Students on the application track spend their prep weeks learning Streamlit, database basics, and the router's behavior at a conceptual level (no deep math required for most).

**A1 — Router Decision Dashboard + Comparative Analysis** (Starter, no GPU)
Build a Streamlit dashboard that visualizes the router's decisions: mode selection probabilities, action scoring, PID dynamics, DAG execution traces. Then use it to conduct a systematic study: mode confusion analysis (when is the router uncertain?), cost-quality Pareto analysis (how does budget affect routing?), and failure pattern identification (where does the router consistently go wrong?).
*Depends on S5 checkpoints at week 6.*

**A2 — Spectral Memory Inspector + Anomaly Detection** (Starter, no GPU)
Build a visual debugger for SharedSpectralMemory showing operator spectra, condition number evolution, and read/write activity. Then add a HealthMonitor with spike detection and spectral collapse detection, three recovery policies (rebuild, ridge boost, selective forget), and a quantitative comparison of which recovery policy works best under different failure scenarios.

**A3 — OfficeQA Demo App + Retrieval Strategy Comparison** (Starter, minimal GPU)
Build a web app where users upload a PDF, watch it get segmented, ask questions, and see retrieved segments with answers. Then implement 3 retrieval baselines (top-k cosine, fixed-chunk, BM25) and run a systematic comparison against SKA's pricing-guided retrieval on 50+ queries, including a lambda sensitivity analysis.

**A4 — Retrieval Quality Feedback Loop** (Intermediate, minimal GPU)
Build a human-in-the-loop feedback system: users rate segments (thumbs up/down) and answers (1–5 stars), feedback is stored in SQLite, converted to training data, used to fine-tune the RewardPredictor, and evaluated via A/B comparison against the original router. You also derive per-mode lambda recommendations from segment-level feedback patterns.
*Depends on S5 checkpoints at week 6.*

---

## Critical

The S5 Router Training project produces trained checkpoints that 4 other projects depend on. S5 delivers v1 checkpoints at week 6 (end of their build week 3). The dependent projects (A1, A4, S4, M4) can build their infrastructure with mock data during weeks 1–6, then integrate the trained router from week 7 onward. All other projects have no dependencies and can start immediately. If no one wants to do this, I can code up this section quickly so it's easy for everyone else to run their code and test. 

---

## Quarter Timeline

| Weeks | Phase | What You Do |
|-------|-------|-------------|
| 1–3 | Prep | Read learning materials. Complete exercises. Build mathematical and tool foundations specific to your project track. Each project guide specifies exactly what to study and what deliverables to produce during prep. |
| 4–10 | Build | Follow the build plan in your project guide. One deliverable per week. Final report due week 10. |

Each project guide contains:

- **Starting requirements** — mathematical and programming prerequisites
- **3 prep weeks** — study tasks and deliverables to build foundations
- **7 build weeks** — implementation tasks with expected outputs
- **"What done looks like"** — concrete checklist of final deliverables

---

## Project Tiers and GPU Requirements

| Tier | Description | Projects |
|------|-------------|----------|
| **Starter** | Good first contact with the codebase. Working prototype early, second half is analysis. | A1, A2, A3 |
| **Intermediate** | Requires understanding one subsystem well. | M1, M2, M4, S2, S4, S5, A4 |
| **Advanced** | Requires deep spectral theory or distributed systems knowledge. | M3, S1, S3 |

| GPU Requirement | Projects |
|-----------------|----------|
| **None** | A1, A2, S1, S3, S5 |
| **Minimal (CPU or brief Colab)** | A3, A4, M2, M4, S2, S4 |
| **~1–2 hours one-time** | M1, M3 |

---

## How to Work on a Project

1. **Weeks 1–3 (Prep):** Read your project guide's Starting Requirements and Prep Weeks sections. Complete the study tasks and produce the prep deliverables. Don't skip this — students who rush into code without understanding the math or tools spend more time debugging than building.
2. **Week 4 (Build Week 1):** Start implementing. Your project guide specifies exactly what to build and what the expected output looks like each week.
3. **Follow the weekly plan** — each week builds on the previous week's output. If you fall behind, the guide's structure helps you prioritize what matters most.
4. **The `ska_agent/` codebase** provides all the core modules you need. Your project guide lists the specific files you'll interact with — read those files before writing your own code.

# CS197/CS197C: Spectral Koopman Attention — Student Projects
**DO NOT SHARE OUTSIDE OF CS197/CS197C SPRING 2026 WITHOUT PERMISSION FROM ALEXANDER JOHANSEN**

This repository contains learning materials and 13 independent research projects exploring **Spectral Koopman Attention (SKA)** — a method that replaces standard transformer attention with Koopman operators. Instead of storing a key-value pair for every token (O(T²) compute, O(T·d) memory), SKA compresses the entire input into a small, fixed-size matrix (typically 64×64, ~32KB) that captures the same temporal dynamics **independent of context length**. At 256K tokens, standard attention needs ~256MB of KV cache per layer; SKA needs 32KB — an 8,000× reduction.

You don't need to understand all of that yet. The [Technical Background](#technical-background) section at the end of this document explains the math in detail. Start here to figure out **what you'll be doing and how to pick a project**.

### Quick Start

1. Follow the [Setup Guide](SETUP.md) (~5 minutes).
2. Read the [Quarter Timeline](#quarter-timeline) below to understand the schedule.
3. Skim the [Skill Requirements](#skill-requirements-at-a-glance) table and pick a project that fits your background.
4. Go to your project's folder under [`ska_projects/`](#repository-structure) and read its `README.md`.
5. Start the [Week 1 learning materials](learning/week1_koopman_theory.md).

If you encounter an unfamiliar term anywhere in the project docs, check the [Glossary](GLOSSARY.md).

---

## Quarter Timeline

| Weeks | Phase | What You Do |
|-------|-------|-------------|
| 1–3 | **Prep** | Read learning materials. Complete exercises. Build mathematical and tool foundations for your project track. |
| 4–10 | **Build** | Follow the build plan in your project guide. One deliverable per week. Final report due week 10. |

Each project guide contains:
- **Starting requirements** — mathematical and programming prerequisites
- **3 prep weeks** — study tasks and deliverables to build foundations
- **7 build weeks** — implementation tasks with expected outputs
- **"What done looks like"** — concrete checklist of final deliverables

---

## Choosing a Project

### Difficulty Tiers

| Tier | Description | Projects |
|------|-------------|----------|
| **Starter** | Good first contact with the codebase. Working prototype early, second half is analysis. | A1, A2, A3 |
| **Intermediate** | Requires understanding one subsystem well. | M1, M2, M4, S2, S4, S5, A4 |
| **Advanced** | Requires deep spectral theory or distributed systems knowledge. | M3, S1, S3 |

### GPU Requirements

| GPU Requirement | Projects |
|-----------------|----------|
| **None** | A1, A2, S1, S3, S5 |
| **Minimal (CPU or brief Colab)** | A3, A4, M2, M4, S2, S4 |
| **~1–2 hours one-time** | M1, M3 |

### Skill Requirements at a Glance

Use this table to find projects that match your background. Each project description below includes a detailed **SKILL REQ** line.

| Skill | Required For | Helpful For |
|-------|-------------|-------------|
| **Linear algebra** (SVD, factorizations, eigenvalues) | M1, M3, S1 | M2, M4, A2 |
| **Multivariable calculus** (gradients, optimization) | M3 | M2, S4 |
| **Probability & statistics** | M4, S5 | M2, A4 |
| **Machine learning fundamentals** (training loops, MLPs, losses) | M2, S5 | M4, A4 |
| **Control theory** (PID, feedback loops) | S4 | — |
| **Algorithms & data structures** (DP, graphs, buffers) | S2, S3 | S1 |
| **TypeScript / async programming** | S3 | — |
| **Python + Streamlit** (web apps, visualization) | A1, A2, A3, A4 | All others |
| **Databases** (SQL, schema design) | A4 | — |
| **Numerical methods** (stability, conditioning, floating-point) | S1 | M1, M3 |

**No math background?** Start with **A1**, **A2**, or **A3** — these are designed for students who are strong programmers but new to engineering math. You will learn to interpret the math through building visualizations, not through derivations.

**Strong math, less programming?** The Math track projects (**M1–M4**) are analysis-heavy and the code is primarily NumPy/PyTorch scripts, not large software systems. **M4** has the lightest programming load.

---

## The 13 Projects

### Math/Theory Track

These projects investigate the mathematical properties of the Koopman operator, the regularization landscape, and the learned components of the routing system. Students on the math track build foundations in linear algebra, spectral theory, and optimization during the prep weeks before running empirical studies.

**M1 — [SVD Initialization Ablation for SKA Surgery](ska_projects/statml/svd_init_ablation/)** (Intermediate, ~1hr GPU)
The SKA surgery process replaces attention layers with Koopman operators, initializing the key projections from the original attention weights via SVD. But sqrt-Σ scaling is just one of many possible strategies. You implement 7 initialization strategies (random orthogonal, random Gaussian, SVD with three different scalings, PCA, NMF), analyze their spectral properties (Gram matrix condition numbers, spectral gaps), run small-scale training comparisons across multiple ranks, and produce a data-backed recommendation for the default strategy.
**SKILL REQ:** Linear algebra (SVD, matrix factorizations, rank truncation, condition numbers). Python proficiency (NumPy, SciPy). Some PyTorch exposure for weight extraction. Comfort reading and analyzing spectral properties of matrices.

**M2 — [Neural Sparsity Parameter Prediction](ska_projects/statml/learned_lambda/)** (Intermediate, minimal GPU)
The PricingEngine uses a fixed lambda to control the sparsity-quality tradeoff in retrieval. You train a small MLP that predicts the optimal lambda per query — high lambda for simple lookups (few segments), low lambda for complex multi-hop queries (thorough retrieval). You generate training data by sweeping lambda values, train and evaluate the predictor against fixed-lambda baselines and an oracle, analyze what the model learns about query complexity, and test cross-document robustness.
**SKILL REQ:** Optimization basics (loss functions, gradient descent, Lagrange duality is a plus). Intro probability/statistics (train/val splits, distributions). Machine learning fundamentals (MLP architecture, training loops). Python proficiency (PyTorch, NumPy).

**M3 — [Spectral and Orthogonal Regularization Interaction](ska_projects/statml/regularization_analysis/)** (Advanced, ~2hrs GPU)
Two regularization losses govern operator health during training: spectral regularization (penalizing ||A_w||_F², which controls operator strength) and orthogonal regularization (penalizing ||W_K^T W_K − I||, which controls input conditioning). You run a 25-point grid sweep over their joint space, produce heatmaps of 5 health metrics, identify Pareto-optimal configurations with sensitivity analysis, characterize the behavior at the four extreme corners, and recommend default parameters.
**SKILL REQ:** Linear algebra (matrix norms, orthogonality, singular values, spectral radius). Multivariable calculus (gradient-based optimization, regularization as penalty terms). Multi-objective optimization (Pareto frontiers). Python proficiency (PyTorch, NumPy). Data visualization (heatmaps, Pareto plots).

**M4 — [Mode Selector Calibration + Calibrated Routing](ska_projects/statml/mode_calibration/)** (Intermediate, minimal GPU)
Neural networks are notoriously overconfident. If the ModeSelector says "95% LOOKUP" but is only right 60% of the time, the router trusts it too much and never explores alternatives. You measure calibration (ECE, reliability diagrams), apply three post-hoc calibration methods (temperature scaling, Platt scaling, histogram binning), then build an exploration policy that uses calibrated confidence to decide when to try multiple modes.
**SKILL REQ:** Probability (softmax, calibration, confidence intervals). Statistics (binning, histograms, Expected Calibration Error). Intro machine learning (classification, logistic regression). Python proficiency (NumPy, SciPy, PyTorch). Data visualization (reliability diagrams).
*Depends on S5 checkpoints at week 6.*

### Systems/Engineering Track

These projects build or optimize core infrastructure components. Students on the systems track learn tools (TypeScript, NumPy internals) and algorithms (Cholesky updates, topological sort, PID control) during the prep weeks before writing production code.

**S1 — [Incremental Cholesky Operator Updates](ska_projects/systems/incremental_cholesky/)** (Advanced, no GPU)
Pure numerical linear algebra. The shared spectral memory rebuilds the Koopman operator from scratch (O(r³)) on every write. You implement O(r²)-per-step incremental updates using rank-1 Cholesky updates with Givens rotations, propagate those rotations to the cached whitened operator, track numerical stability over 100K+ updates, implement refactorization scheduling, and benchmark speedups at ranks 32–512.
**SKILL REQ:** Linear algebra (Cholesky factorization, triangular solves, rank-1 updates, Givens rotations). Numerical methods (floating-point error analysis, condition number drift, computational complexity). This is the most math-heavy systems project. Python proficiency (NumPy, SciPy).

**S2 — [Streaming Bounded-Memory Segmentation](ska_projects/systems/streaming_segmentation/)** (Intermediate, minimal GPU)
The geometry learner's batch DP loads all sentences before segmenting. You build a streaming version that processes sentences one at a time with O(lookback_k) memory, produces identical segments to the batch algorithm, and emits finalized segments incrementally. This enables the offline pipeline to process arbitrarily large corpora without running out of memory.
**SKILL REQ:** Algorithms (dynamic programming, sliding window techniques). Data structures (circular buffers, deques). Python proficiency (NumPy). Comfort with memory profiling and performance benchmarking. No heavy math required — this is primarily an algorithms and systems project.

**S3 — [TypeScript Orchestrator Integration](ska_projects/systems/ts_orchestrator/)** (Advanced, no GPU)
The Python ToolServer has endpoints for reasoning, retrieval, and code execution, but no client. You build a TypeScript orchestrator that decomposes goals into task DAGs, uses topological sort (Kahn's algorithm) to identify parallel groups, dispatches tasks via HTTP with retries and timeouts, synchronizes shared memory, and handles partial failures gracefully. Team project (2–3 students).
**SKILL REQ:** TypeScript/JavaScript (async/await, Promises, type system). Graph algorithms (DAGs, topological sort). Networking (HTTP/REST, JSON APIs, error handling, retries). No math beyond basic graph theory — this is a pure software engineering project.

**S4 — [PID Controller Auto-Tuning](ska_projects/systems/pid_autotuning/)** (Intermediate, minimal GPU)
The PID controller's gains (Kp, Ki, Kd) are hand-tuned. You build an automated system: grid search over the 3D gain space, Bayesian optimization for sample-efficient search, the Ziegler-Nichols heuristic from control theory, and budget scheduling policies (constant, front-loaded, cosine). You compare all methods and recommend optimal gains with a specific budget schedule.
**SKILL REQ:** Control theory basics (PID equations, feedback loops, stability). Optimization (grid search, Bayesian optimization via Gaussian processes). Multivariable calculus is helpful but not essential. Python proficiency (NumPy, scikit-optimize).
*Depends on S5 checkpoints at week 6.*

**S5 — [Router Training Pipeline](ska_projects/systems/router_training/)** (Intermediate, no GPU) **CRITICAL**
Four other projects (A1, A4, S4, M4) depend on your trained checkpoints. You build the data generation and training pipeline for the ModeSelector (4-way query classifier) and RewardPredictor (marginal quality estimator). Person A trains the ModeSelector on 1000+ template-generated queries. Person B trains the RewardPredictor on specialist quality measurements. **v1 checkpoints must ship by end of week 6.** Team project (2 students).
**SKILL REQ:** Machine learning fundamentals (classification, regression, cross-entropy loss, train/val/test splits, confusion matrices). Intro probability (softmax, confidence scores). Python proficiency (PyTorch, NumPy). Comfort with dataset generation and training pipelines.

### Application Track

These projects build user-facing tools and conduct empirical analyses. Students on the application track learn Streamlit, database basics, and the router's behavior at a conceptual level during the prep weeks (no deep math required for most).

**A1 — [Router Decision Dashboard + Comparative Analysis](ska_projects/applied/router_dashboard/)** (Starter, no GPU)
Build a Streamlit dashboard that visualizes the router's decisions: mode selection probabilities, action scoring, PID dynamics, DAG execution traces. Then use it to conduct a systematic study: mode confusion analysis (when is the router uncertain?), cost-quality Pareto analysis (how does budget affect routing?), and failure pattern identification (where does the router consistently go wrong?).
**SKILL REQ:** Python proficiency (Streamlit, Plotly). No math prerequisites — you will *learn* to interpret probabilities and cost-quality tradeoffs through the project. Good fit if you are stronger in programming and data visualization than in math.
*Depends on S5 checkpoints at week 6.*

**A2 — [Spectral Memory Inspector + Anomaly Detection](ska_projects/applied/memory_inspector/)** (Starter, no GPU)
Build a visual debugger for SharedSpectralMemory showing operator spectra, condition number evolution, and read/write activity. Then add a HealthMonitor with spike detection and spectral collapse detection, three recovery policies (rebuild, ridge boost, selective forget), and a quantitative comparison of which recovery policy works best under different failure scenarios.
**SKILL REQ:** Python proficiency (Streamlit, Plotly, NumPy). No math prerequisites — you will learn to interpret eigenvalues and condition numbers through the project, but you don't need to derive them. Good fit for students interested in monitoring/observability systems.

**A3 — [OfficeQA Demo App + Retrieval Strategy Comparison](ska_projects/applied/officeqa_demo/)** (Starter, minimal GPU)
Build a web app where users upload a PDF, watch it get segmented, ask questions, and see retrieved segments with answers. Then implement 3 retrieval baselines (top-k cosine, fixed-chunk, BM25) and run a systematic comparison against SKA's pricing-guided retrieval on 50+ queries, including a lambda sensitivity analysis.
**SKILL REQ:** Python proficiency (Streamlit, sentence-transformers, pdfplumber). No math prerequisites. Familiarity with information retrieval concepts (similarity search, embeddings) is helpful but will be taught. The most accessible project for students new to ML research.

**A4 — [Retrieval Quality Feedback Loop](ska_projects/applied/retrieval_feedback/)** (Intermediate, minimal GPU)
Build a human-in-the-loop feedback system: users rate segments (thumbs up/down) and answers (1–5 stars), feedback is stored in SQLite, converted to training data, used to fine-tune the RewardPredictor, and evaluated via A/B comparison against the original router. You also derive per-mode lambda recommendations from segment-level feedback patterns.
**SKILL REQ:** Python proficiency (Streamlit, SQLite, NumPy). Basic statistics (rating normalization, A/B comparison). Database design (schema, queries). Intro machine learning (fine-tuning, training data). No heavy math — the challenge is building a clean end-to-end data pipeline.
*Depends on S5 checkpoints at week 6.*

---

## Project Dependencies

The **S5 Router Training** project produces trained checkpoints that 4 other projects depend on. S5 delivers v1 checkpoints at week 6 (end of their build week 3). The dependent projects (**A1, A4, S4, M4**) can build their infrastructure with mock data during weeks 1–6, then integrate the trained router from week 7 onward. All other projects have no dependencies and can start immediately.

If no one picks S5, I will build it myself so everyone else can run their code and test.

---

## Setup

See the **[Setup Guide](SETUP.md)** for full instructions (virtual environment, dependencies, verification, troubleshooting).

Quick version:

```bash
cd cs195-ska-projects
python3 -m venv venv && source venv/bin/activate
pip install torch numpy scipy matplotlib transformers sentence-transformers
pip install streamlit plotly scikit-optimize rank_bm25 datasets
```

Verify: `python3 -c "from shared.ska import SKAModule; print('OK')"`

All projects reference the `shared/ska_agent-1.0.0-8/ska_agent/` codebase. The project guide files in each project's folder explain which modules you'll interact with.

---

## Repository Structure

```
cs195-ska-projects/
├── SETUP.md                           How to install and verify your environment
├── GLOSSARY.md                        Definitions of all technical terms used in project docs
│
├── learning/                          Week 1–2 learning materials (start here)
│   ├── week1_koopman_theory.md            Koopman operator theory from first principles
│   └── week2_hands_on.md                  Hands-on exercises with SKA and Koopman MLP
│
├── shared/                            Shared code used by all projects
│   ├── ska.py                             Standalone SKA module (for learning/exercises)
│   ├── koopman_mlp.py                     Standalone Koopman MLP (for learning/exercises)
│   ├── eval_tasks.py                      Evaluation benchmarks (MQAR)
│   ├── utils.py                           Shared utilities
│   └── ska_agent-1.0.0-8/ska_agent/       The full SKA-Agent library
│       ├── pipeline.py                        End-to-end pipeline wiring
│       ├── core/                              SKA module, geometry learner, pricing engine
│       ├── models/                            Jamba+SKA surgery, Qwen coordinator, embedder
│       ├── router/                            Adaptive router, PID controller
│       ├── shared_memory/                     Spectral memory, Think-Koopman bridge
│       ├── training/                          SKA trainer, router trainer
│       ├── evaluation/                        OfficeQA evaluation
│       └── orchestration/                     TypeScript integration layer (ToolServer)
│
└── ska_projects/                      The 13 projects (each has README, SPEC,
    │                                  BACKGROUND, STARTER, and EVALUATION docs)
    │
    ├── statml/                        --- Math/Theory Track (M) ---
    │   ├── svd_init_ablation/             M1: Compare 7 initialization strategies
    │   ├── learned_lambda/                M2: Neural sparsity parameter prediction
    │   ├── regularization_analysis/       M3: Spectral + orthogonal regularization
    │   └── mode_calibration/              M4: ModeSelector calibration + exploration
    │
    ├── systems/                       --- Systems/Engineering Track (S) ---
    │   ├── incremental_cholesky/          S1: O(r²) incremental operator updates
    │   ├── streaming_segmentation/        S2: Streaming bounded-memory DP segmentation
    │   ├── ts_orchestrator/               S3: TypeScript orchestrator + parallel DAGs
    │   ├── pid_autotuning/                S4: PID cost controller gain optimization
    │   └── router_training/               S5: Router training pipeline (CRITICAL)
    │
    └── applied/                       --- Application Track (A) ---
        ├── router_dashboard/              A1: Router decision visualization + analysis
        ├── memory_inspector/              A2: Spectral memory debugger + anomaly detection
        ├── officeqa_demo/                 A3: Document QA + retrieval comparison
        └── retrieval_feedback/            A4: Human feedback loop for router improvement
```

Each project folder contains five docs:
- **README.md** — project overview, motivation, and what you'll build
- **SPEC.md** — detailed weekly build plan with expected outputs
- **BACKGROUND.md** — technical context specific to this project
- **STARTER.md** — starter code or minimal working examples
- **EVALUATION.md** — grading rubric and deliverables checklist

---

## How to Work on a Project

1. **Weeks 1–3 (Prep):** Read your project guide's Starting Requirements and Prep Weeks sections. Complete the study tasks and produce the prep deliverables. Don't skip this — students who rush into code without understanding the math or tools spend more time debugging than building.
2. **Week 4 (Build Week 1):** Start implementing. Your project guide specifies exactly what to build and what the expected output looks like each week.
3. **Follow the weekly plan** — each week builds on the previous week's output. If you fall behind, the guide's structure helps you prioritize what matters most.
4. **The `shared/ska_agent-1.0.0-8/ska_agent/` codebase** provides all the core modules you need. Your project's SPEC.md lists the specific files you'll interact with — read those files before writing your own code.

---

## Technical Background

> The rest of this document is reference material. You don't need to read it all before choosing a project — your project guide will tell you what's relevant. Come back here when you want to understand how the pieces fit together.

### The Core Idea: Koopman Operators for Attention

#### What Is a Koopman Operator?

The Koopman operator comes from dynamical systems theory. Given a nonlinear dynamical system where states evolve over time, the Koopman operator is an infinite-dimensional linear operator that exactly represents the dynamics — but in a lifted function space rather than the original state space. The key insight is that **nonlinear dynamics become linear** when viewed through the right lens.

For sequence modeling, we treat consecutive token representations as a discrete dynamical system: token t evolves into token t+1 through some (nonlinear) function of the transformer's hidden states. The Koopman operator approximation captures this evolution as a matrix multiply — and because it's linear, we can analyze its spectral properties (eigenvalues, singular values) to understand what the sequence is doing.

#### How SKA Uses This

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

### The Koopman MLP: A Simpler Variant

Not every project uses the full SKA attention replacement. Several projects work with the **Koopman MLP**, a simpler module that applies the same Koopman operator idea to feedforward layers:

1. Project a hidden state into a low-rank space
2. Apply a learned Koopman matrix (with spectral normalization)
3. Project back to the original dimension

This lets you study the Koopman operator's properties (spectral structure, initialization, regularization) without the complexity of the full attention replacement. The math-track projects (M1–M4) investigate these properties in depth.

### SKA-Agent: The Full System Built on SKA

The projects in this repository study SKA as a standalone attention mechanism, but SKA was designed as the foundation for a larger system: **SKA-Agent**, an adaptive multi-model orchestration framework where multiple specialist models collaborate on complex queries, communicating through shared Koopman operators instead of passing text back and forth.

#### The Problem SKA-Agent Solves

Consider a question like: *"Compare defense spending as a percentage of GDP across the last three fiscal years, calculate the average annual growth rate, and explain the primary drivers."* This requires three different capabilities: document retrieval (finding the right numbers in budget reports), computation (calculating growth rates), and reasoning (explaining the drivers). No single model is optimal for all three — a small retrieval specialist is fast and cheap for finding facts, a code executor handles calculations exactly, and a large reasoning model synthesizes the explanation.

The naive approach is to dump everything into one model's context window: paste the retrieved documents, the calculation results, and the original question into GPT-4 and hope for the best. This works, but it's expensive (the context window fills up fast), brittle (irrelevant retrieved passages dilute the signal), and opaque (you can't see which component contributed what).

SKA-Agent takes a different approach. It routes queries to functionally distinct specialists — a retriever, a code executor, a reasoner — and the specialists share what they learn through **spectral memory**: fixed-size Koopman operators that compress each specialist's findings into a small matrix that other specialists can query.

#### The Three-Tier Architecture

SKA-Agent uses three tiers of models, each with different cost and capability profiles:

**Tier 1 — Qwen Coordinator (27B, quantized to ~16.5GB):** The primary reasoner and coordinator. It handles task decomposition (breaking a complex goal into subtasks), structured reasoning (with `<think>` chain-of-thought traces), and final answer synthesis. Its reasoning traces are projected into the shared spectral memory via the Think-Koopman Bridge, so other agents can query what the coordinator is thinking about without reading the full text of its reasoning chain.

**Tier 2 — Jamba+SKA (12B active, retrieval specialist):** A Jamba model where 4 attention layers (at positions 4, 12, 20, 28) have been surgically replaced with SKA modules. This is the retrieval specialist: it ingests documents through the SKA operator and produces dense, compressed representations that the pricing engine can query. The SKA surgery process extracts the pretrained attention weights, runs truncated SVD to find the top-r directions, and initializes the SKA projections with sqrt-singular-value scaling — preserving the model's learned representations while fitting them into the fixed-size operator framework.

**Tier 3 — DeepSeek V3 (optional, for hard multi-hop reasoning):** A heavy reasoning model invoked only for queries that exceed the coordinator's capabilities. Cost-gated by the PID controller so it's only called when the expected quality improvement justifies the expense.

#### How Specialists Communicate: Shared Spectral Memory

This is the architectural innovation that makes SKA-Agent more than just "a router that picks which model to call." In a conventional multi-agent system, Agent A writes 2,000 tokens of findings into Agent B's context window — growing B's KV cache by ~32MB per handoff and consuming precious context space. In SKA-Agent, Agent A writes key vectors that update a rank-r Koopman operator (~32KB), and Agent B queries that operator through the power spectral filter to retrieve what A learned.

The shared memory has multiple slots organized as a multi-head Koopman module (K=4 parallel rank-48 operators per attention head):

- **Slots 0–1:** Document structure (written by the parser specialist)
- **Slot 2:** Reasoning state (written by the coordinator via the Think-Koopman Bridge, which projects the coordinator's `<think>` hidden states into operator space)
- **Slot 3:** Temporal patterns

Each slot has its own key projection, Gram matrix, and gate. Slot specialization emerges during training — different agents naturally write to different slots because their key distributions are different, and the per-slot gates learn to weight the slots appropriately for each query type.

The operator captures temporal dynamics: the transition matrix M = Σ z_{t+1} zₜᵀ encodes how an agent's findings evolved across steps. Directions where reasoning was consistent (high eigenvalues) are amplified by the power filter; directions that fluctuated randomly (low eigenvalues) are suppressed. So the operator doesn't just store *what* an agent concluded — it stores *how confident* that conclusion is, encoded in the spectral structure.

#### The Router: Sequential Marginal Evaluation

The router decides which specialists to invoke and in what order. It has four learned components:

**QueryEncoder** (frozen MiniLM-L6-v2, 22M params): Encodes the input query into a 384-dim dense vector. Not trained — it's a pretrained sentence transformer used as a fixed feature extractor.

**ModeSelector** (2-layer MLP, ~100K params): Classifies the query into one of four collaboration modes — LOOKUP (simple factual question), MULTI_DOC (cross-document comparison), COMPUTE (calculation needed), MULTI_STEP (complex multi-part reasoning). Each mode defines a DAG template of valid transitions between specialist types.

**RewardPredictor** (3-layer MLP, ~400K params): Predicts the quality improvement from choosing specialist m over a baseline: Δr̂(a) = g(e_Q, e_m − e_base). This is the marginal value estimate: "how much better will the answer be if we invoke the retriever vs. just using the parser?"

**PID Controller** (no learned params): Adjusts a 5-dimensional price vector λ = (λ_in, λ_out, λ_lat, λ_$, λ_meta) based on the running cost rate. The five dimensions correspond to input tokens, output tokens, latency, dollar cost, and shared memory overhead.

The decision rule at each step: enumerate candidate actions from the current DAG node, score each as S(a) = Δr̂(a) − λᵀ · Δĉ(a), and execute the highest-scoring action if its score is positive. If no action has positive score, stop. The execution graph isn't predicted up front — it emerges from a sequence of local positive-score decisions. Simple queries terminate after 1–2 actions; complex multi-hop queries chain 4–5 specialists.

#### The Retrieval Pipeline: Geometry Learning + Pricing-Guided Selection

Before any model inference happens, documents are preprocessed through a two-stage pipeline:

**Stage I — Geometry Learning:** Raw text is split into sentences, embedded, and segmented using dynamic programming. Unlike fixed-size chunking (every 512 tokens), the DP finds natural topic boundaries by minimizing internal pairwise cosine distance within each segment, subject to a sparsity penalty λ per segment. The result is semantically coherent atomic units that respect the document's natural structure. The DP runs in O(N·K) time where N is the sentence count and K is the lookback window (default 50).

**Stage II — Pricing-Guided Retrieval:** Given a query, the PricingEngine selects segments using a greedy optimization: the reduced cost for adding segment j is c̄(j) = λ + η·redundancy − information_gain(j), where the information gain is the Schur complement of the query-segment projection. A segment is included only if c̄(j) < 0 — meaning its information gain exceeds the cost threshold. After each selection, the query residual is updated by orthogonal projection (removing the explained component), which automatically prevents redundancy: once a segment's information is "used up," subsequent segments covering the same topic have near-zero information gain.

#### How Projects Connect to SKA-Agent

Every project in this repository explores a piece of this larger system:

- **S1 and M1** directly implement and study core SKA-Agent components (incremental Cholesky updates for streaming memory, initialization strategies for the Jamba surgery process).
- **M2, M3, and M4** study the mathematical properties (sparsity prediction, regularization landscape, calibration) that determine whether the routing and retrieval systems work reliably.
- **S2, S3, S4, and S5** build or optimize infrastructure (streaming segmentation, TypeScript orchestrator, PID tuning, router training) that the full system depends on.
- **A1, A2, A3, and A4** build user-facing tools (dashboards, demo apps, feedback loops) that make the system visible, testable, and improvable.

Understanding SKA-Agent gives you context for *why* your project matters: you're not studying a matrix factorization in isolation — you're studying a component of a system where multiple models collaborate through spectral memory, and the reliability of that collaboration depends on the mathematical properties you're investigating.

### The Math You Need to Know

The first two weeks of the course build up the mathematical foundations. Here's what you'll learn and why it matters:

**Singular Value Decomposition (SVD):** Every matrix A can be written as A = UΣVᵀ where U and V are orthogonal and Σ is diagonal with non-negative entries (the singular values). The singular values tell you the "strength" of the matrix in different directions. In SKA, SVD is used for: (a) initializing key/query projections from pretrained attention weights (M1), (b) spectral normalization of the operator, and (c) analyzing what the operator has learned (M3).

**Cholesky Factorization:** A symmetric positive definite matrix G can be uniquely factored as G = LLᵀ where L is lower triangular. This is the numerically stable way to "invert" the Gram matrix — instead of computing G⁻¹ explicitly (which amplifies errors), we solve triangular systems Lx = b, which is O(r²) and backward-stable. Every operator construction in SKA goes through Cholesky. S1 extends this to incremental rank-1 updates.

**Condition Number:** κ(G) = σ_max/σ_min measures how sensitive the solution of Gx = b is to perturbations in G. When κ is large, the Cholesky factorization amplifies numerical errors, and the whitened operator A_w becomes unreliable. The ridge regularization εI in the Gram matrix keeps κ bounded — this is the same idea as Tikhonov regularization in inverse problems.

**Power Iteration and Spectral Filtering:** Repeatedly multiplying a vector by a matrix A amplifies the components aligned with A's dominant eigenvectors and suppresses the rest. After K applications, the output is dominated by the top-K eigenspaces. In SKA, this is how the query "reads" the prefix: the power filter A_w^K amplifies temporally consistent patterns and suppresses noise.

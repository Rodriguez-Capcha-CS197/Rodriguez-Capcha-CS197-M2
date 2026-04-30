# Glossary

Quick reference for technical terms used across the project guides. Terms are alphabetical. If you're looking for a deeper explanation, see the [Technical Background](README.md#technical-background) section of the main README or the learning materials in `learning/`.

---

**Amortized cost** — The average cost per operation when an expensive operation (e.g., full matrix refactorization at O(r³)) happens infrequently alongside many cheap operations (e.g., rank-1 updates at O(r²)).

**Amortized optimization** — Learning a function that directly predicts the solution to an optimization problem for a given input, rather than solving the optimization from scratch each time. Used in M2 to predict optimal lambda per query.

**Bayesian optimization** — A sample-efficient optimization method that builds a probabilistic model (typically a Gaussian process) of the objective function and uses it to decide where to evaluate next. Used in S4 for PID gain tuning.

**Binning** — Partitioning predictions into groups by confidence range (e.g., [0.0–0.1), [0.1–0.2), ..., [0.9–1.0]) to measure how well predicted probabilities match actual outcomes. Used in M4 for calibration analysis.

**Calibration (probabilistic)** — A classifier is calibrated when its predicted probabilities match empirical frequencies. If a model says "80% confident" on 100 predictions, roughly 80 of those should be correct. Measured by ECE. See M4.

**Cholesky factorization** — The unique decomposition of a symmetric positive definite matrix G into G = LLᵀ, where L is lower triangular. Used throughout SKA to stably "invert" the Gram matrix via triangular solves instead of explicit inversion. See also: SPD.

**Circular buffer** — A fixed-size array that wraps around: when the end is reached, new elements overwrite the oldest. Enables bounded-memory streaming algorithms. Used in S2.

**Condition number (κ)** — κ(G) = σ_max / σ_min, the ratio of the largest to smallest singular value of a matrix. Measures sensitivity to numerical errors. κ ≈ 1 is well-conditioned (stable); κ > 10⁵ is ill-conditioned (unreliable). Ridge regularization keeps κ bounded.

**Cosine distance** — A distance measure between vectors: 1 − (u · v) / (‖u‖ ‖v‖). Equals 0 when vectors point the same direction, 1 when orthogonal, 2 when opposite. Used in geometry learning for segment similarity.

**DAG (Directed Acyclic Graph)** — A graph with directed edges and no cycles. Allows a unique topological ordering. Used in S3 to represent task dependencies, and in the router to define valid specialist transitions.

**Dense vector / Embedding** — A fixed-length array of floating-point numbers that represents the semantic meaning of text. Produced by an encoder model (e.g., MiniLM). Similar texts have similar embeddings (high cosine similarity).

**Eckart-Young theorem** — States that truncated SVD gives the best rank-r approximation of a matrix in both Frobenius and spectral norm. Justifies using SVD for dimensionality reduction and initialization. Referenced in M1.

**ECE (Expected Calibration Error)** — A metric for calibration quality: the weighted average of |accuracy − confidence| across probability bins. ECE = 0 means perfect calibration. See M4.

**Eigenvalue / Eigenvector** — For a square matrix A, a scalar λ and nonzero vector v satisfying Av = λv. The eigenvalue λ tells you how much A stretches in the direction v. In SKA, eigenvalues of the Koopman operator indicate the strength of temporal patterns.

**Explicit feedback** — Direct user input about quality: thumbs up/down, star ratings, binary accept/reject. Contrast with implicit feedback. See A4.

**Frobenius norm (‖·‖_F)** — The square root of the sum of all squared elements of a matrix: ‖A‖_F = √(Σᵢⱼ aᵢⱼ²). Equivalently, √(Σ σᵢ²) where σᵢ are the singular values. A natural measure of matrix "size."

**Gaussian process** — A probabilistic model that defines a distribution over functions. Provides both a prediction and an uncertainty estimate at every point. Used in Bayesian optimization (S4) to model the objective surface.

**Givens rotation** — A sparse orthogonal matrix that rotates a vector in a single 2D plane, used to zero out one element at a time. The building block for incremental Cholesky updates. See S1.

**Gram matrix** — G = XᵀX (or Σ zₜzₜᵀ + εI in SKA), a matrix whose entries are inner products between data vectors. Captures pairwise similarities and the overall distribution shape of the data.

**Implicit feedback** — Inferred user preference from behavior (clicks, dwell time, query reformulations) without the user explicitly stating a judgment. Contrast with explicit feedback. See A4.

**In-degree** — The number of incoming edges to a vertex in a directed graph. In topological sort (Kahn's algorithm), vertices with in-degree 0 have no unresolved dependencies and can execute immediately.

**Information gain** — The reduction in uncertainty from incorporating a new piece of information. In the pricing engine, it measures how much a candidate segment explains the query that hasn't already been explained by previously selected segments.

**Isometry** — A linear transformation that preserves distances: ‖Wx‖ = ‖x‖ for all x. Equivalent to W being an orthogonal matrix (WᵀW = I). Orthogonal regularization pushes weight matrices toward being isometries. See M3.

**Kahn's algorithm** — A BFS-based algorithm for topological sorting: repeatedly remove vertices with in-degree 0, decrement neighbors' in-degrees, repeat. Used in S3 to schedule parallel task groups.

**Koopman operator** — An (infinite-dimensional) linear operator from dynamical systems theory that exactly represents nonlinear dynamics in a lifted function space. In SKA, approximated as a finite r×r matrix that captures how token representations evolve across a sequence. See the [Technical Background](README.md#the-core-idea-koopman-operators-for-attention).

**Lagrangian** — An objective function that incorporates constraints as weighted penalty terms: L(x, λ) = f(x) + λ · g(x). The weight λ (Lagrange multiplier) controls the tradeoff between the objective f and the constraint g. See M2.

**Logits** — The raw, pre-softmax outputs of a neural network classifier. Unnormalized scores where higher values indicate greater confidence in that class. Applying softmax converts logits to probabilities.

**Orthogonal matrix** — A square matrix Q where QᵀQ = I (its transpose is its inverse). Orthogonal matrices preserve vector lengths and angles. Used in SVD (U, V are orthogonal) and as targets for orthogonal regularization.

**Pareto frontier** — The set of solutions where no objective can be improved without worsening another. Points on the frontier represent optimal tradeoffs (e.g., between cost and quality). Used in M3, S4, A1.

**PCA (Principal Component Analysis)** — A dimensionality reduction technique that finds orthogonal directions of maximum variance in the data. Equivalent to SVD on mean-centered data. Referenced in M1 as an initialization strategy.

**PID controller** — A feedback controller with three terms: Proportional (reacts to current error), Integral (reacts to accumulated error), Derivative (reacts to rate of change). Used in SKA-Agent to adjust cost thresholds dynamically. See S4.

**Platt scaling** — A post-hoc calibration method that learns an affine transformation of logits (az + b) before softmax. More expressive than temperature scaling (which only learns a single scalar). See M4.

**Power iteration** — Repeatedly multiplying a vector by a matrix: v ← Av / ‖Av‖. Converges to the dominant eigenvector. In SKA, the power filter A_w^K amplifies temporally consistent patterns and suppresses noise.

**Prefix sums** — Cumulative sums computed incrementally: prefix[i] = Σ_{t=0}^{i} value[t]. Allow O(1) computation of any subarray sum. Used in S2 for efficient distance calculations in the streaming DP.

**Quality delta (Δr)** — The difference in answer quality between two configurations: Δr = quality_new − quality_baseline. Used as the training signal for the RewardPredictor. See A4, S5.

**Rank-1 update** — Adding an outer product zzᵀ to an existing matrix: G_new = G_old + zzᵀ. Enables incremental updates to the Gram matrix without rebuilding from scratch. See S1.

**Reconstruction error** — The difference between original data and its approximation, typically measured as ‖original − approximation‖_F / ‖original‖_F. Lower is better. Used in M1 to compare initialization strategies.

**Reliability diagram** — A plot of binned accuracy (y-axis) vs. average predicted confidence (x-axis). A perfectly calibrated model follows the y = x diagonal. Points above the diagonal mean the model is underconfident; below means overconfident. See M4.

**Reward predictor (RewardPredictor)** — A neural network that predicts the quality improvement (Δr) from calling a particular specialist model. Guides routing decisions by estimating marginal value. Trained in S5, used in A1, A4.

**Ridge regularization** — Adding εI (a small multiple of the identity matrix) to a matrix before inversion or factorization. Keeps the condition number bounded and prevents numerical instability. Also called Tikhonov regularization.

**Schur complement** — For a block matrix [[A, B], [C, D]], the Schur complement is S = D − CA⁻¹B. It captures the information in D not already explained by A. In the pricing engine, used to measure the marginal information gain of a candidate segment.

**Segment** — A group of consecutive sentences that form a semantically coherent unit. The basic retrieval granularity in SKA-Agent's pipeline, produced by the geometry learner's dynamic programming segmentation.

**Semantic coherence** — When consecutive text elements (sentences, paragraphs) belong to the same topic and have high embedding similarity. The geometry learner optimizes segmentation boundaries to maximize within-segment coherence.

**Shadow price** — The optimal value of a Lagrange multiplier λ*. Represents the marginal value of relaxing a constraint by one unit. In M2, the optimal lambda tells you how aggressively to prune retrieval results for a given query.

**Similarity transform** — A matrix transformation B = XAX⁻¹ that preserves eigenvalues but changes the basis. The whitened Koopman operator A_w = L⁻¹ML⁻ᵀ is a similarity transform of the natural operator MG⁻¹.

**Singular values** — The non-negative values σ₁ ≥ σ₂ ≥ ... ≥ 0 on the diagonal of Σ in the SVD A = UΣVᵀ. They measure the "strength" of the matrix in each direction. The largest singular value is the spectral norm.

**Softplus** — A smooth activation function: softplus(x) = log(1 + eˣ). Maps ℝ → (0, ∞), making it useful for predicting strictly positive values like lambda. A smooth approximation of ReLU.

**SPD (Symmetric Positive Definite)** — A matrix G where G = Gᵀ (symmetric) and xᵀGx > 0 for all nonzero vectors x. SPD matrices have real positive eigenvalues and admit a unique Cholesky factorization. The Gram matrix in SKA is SPD by construction.

**Spectral normalization** — Scaling a matrix so its largest singular value equals a target γ ≤ 1. Ensures stability when the matrix is applied repeatedly (power iteration). Without it, repeated application could cause outputs to explode.

**Spectral radius** — The largest absolute eigenvalue of a matrix: ρ(A) = max|λᵢ|. For the Koopman operator, ρ < 1 ensures the power filter is stable. Related to but distinct from the spectral norm (largest singular value).

**Spectral regularization** — A penalty on the singular values of a matrix, typically ‖A‖_F² = Σ σᵢ². Controls the operator's magnitude during training. See M3.

**Streaming quantile sketch** — A data structure (e.g., t-digest, GK sketch) that estimates percentiles from streaming data in a single pass with bounded memory. Referenced in S2 as a stretch goal for automatic lambda tuning.

**SVD (Singular Value Decomposition)** — The factorization A = UΣVᵀ where U and V are orthogonal matrices and Σ is diagonal with non-negative singular values. The fundamental tool for analyzing matrix structure. See the [Technical Background](README.md#the-math-you-need-to-know).

**Temperature scaling** — A post-hoc calibration method that divides logits by a learned temperature T > 0 before softmax: softmax(z/T). T > 1 makes predictions less confident; T < 1 makes them sharper. See M4.

**Topological sort** — A linear ordering of vertices in a DAG such that for every directed edge (u → v), u appears before v. Determines a valid execution order respecting all dependencies. See S3.

**Training signal** — The ground-truth values used to supervise a learning algorithm. In A4, user ratings are converted into training signals (quality deltas) for fine-tuning the RewardPredictor.

**Truncated SVD** — SVD keeping only the top r singular values and their corresponding vectors, yielding the best rank-r approximation (by the Eckart-Young theorem). Used in M1 for initialization.

**Unit roundoff (u)** — The smallest floating-point value where 1.0 + u ≠ 1.0 in machine arithmetic. Approximately 10⁻¹⁶ for 64-bit (double precision) floats. Determines the fundamental precision limit of numerical algorithms. See S1.

**Whitening transform** — A linear transformation (typically via Cholesky: w = L⁻¹z) that removes correlation from data and normalizes scale, making the covariance structure isotropic. In SKA, queries are whitened before the power filter and unwhitened after.

**Ziegler-Nichols method** — A heuristic for tuning PID controller gains. Increase proportional gain until the system oscillates steadily, measure the critical gain and oscillation period, then set Kp/Ki/Kd using lookup-table ratios. See S4.

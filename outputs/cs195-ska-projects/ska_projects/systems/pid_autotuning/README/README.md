# S4: PID Controller Auto-Tuning — Full Project Guide

## Project Classification: Math/Systems-Hybrid

**Tier:** Intermediate | **GPU:** Minimal | **Team Size:** 1 | **Total Duration:** 10 weeks (3 prep + 7 build)

**Dependencies:** Requires trained router checkpoints from S5 Router Training (delivered at S5's week 3, aligning with your build week 1). Use the prep phase to build tuning infrastructure, then run experiments with the trained router.

---

## Project Summary

You are building an automated system to optimize the PID cost controller's gains (Kp, Ki, Kd) and budget parameters. The PID controller governs how aggressively the router penalizes expensive specialist actions. Currently these gains are hand-tuned. You implement grid search, Bayesian optimization, and the Ziegler-Nichols heuristic, then evaluate them with different budget scheduling policies to find optimal cost-quality tradeoffs.

---

## Motivation: Why This Matters for SKA-Agent

The PID controller is the economic governor of the entire multi-agent system. Every time the router considers invoking a specialist, it computes a score: S(a) = predicted_quality_gain - λ^T · predicted_cost. The price vector λ (five dimensions: input tokens, output tokens, latency, dollar cost, shared memory overhead) determines how aggressively the router penalizes expensive actions. When λ is high, the router is conservative and stops early; when λ is low, it spends freely and chains multiple specialists.

The budget control in SKA-Agent is explicit and continuous: the PID controller dynamically adjusts λ based on the running cost rate, and the router uses λ in every scoring decision. This gives fine-grained, real-time cost control — the router doesn't make binary "use this model or that model" decisions, it continuously modulates how aggressively it penalizes expensive specialists based on how much budget has been spent so far.

But the PID gains (Kp, Ki, Kd) are hand-tuned, and they interact in non-obvious ways. Too-high Kp makes λ oscillate wildly (the router alternates between overspending and underspending). Too-high Ki causes integral windup (λ drifts to maximum and stays there, shutting down all specialist invocation). Too-high Kd amplifies noise in the cost signal. The optimal gains depend on the query distribution and the cost characteristics of the available specialists — they're not universal constants. Your autotuning system finds the right operating point for any deployment configuration, ensuring the router spends its budget efficiently rather than wasting it on oscillations or hoarding it due to over-damping.

---

## Starting Requirements

### Mathematical Prerequisites

- **PID control theory:** A PID controller computes a control signal as the sum of three terms: Proportional (reacts to current error), Integral (eliminates steady-state offset by summing past errors), and Derivative (provides damping by reacting to error change rate). The control law is: u(t) = Kp·e(t) + Ki·Σe + Kd·(e(t) - e(t-1)). You need to understand what each gain does and what happens when each is too high or too low.
- **The cost controller's role:** In the router, the PID controller adjusts a 5-dimensional price vector λ that penalizes costly actions. When cost exceeds the budget rate, the error e = avg_cost - budget_rate is positive, so λ increases, making the router more conservative. When cost is under budget, λ decreases. The dynamics: λ_{t+1} = clip(λ_t + Kp·e + Ki·Σe + Kd·Δe, 0, λ_max).
- **Optimization basics:** Grid search (exhaustive evaluation of a parameter grid), Bayesian optimization (using a Gaussian Process surrogate to guide search efficiently), and the Ziegler-Nichols heuristic (a classical control theory method for setting PID gains based on oscillation characteristics).
- **Pareto optimality:** Different gain settings produce different cost-quality tradeoffs. A Pareto-optimal configuration cannot be improved on quality without worsening cost, or vice versa.
- **Budget scheduling:** Instead of a constant budget rate, you can vary the budget over time. Front-loading (high budget early, low budget later) lets the router gather information before becoming conservative. Cosine annealing provides smooth transitions.

### Programming Prerequisites

- Python with NumPy for numerical work
- Matplotlib/Plotly for Pareto plots, lambda trajectories, and convergence curves
- scikit-optimize (`skopt`) for Bayesian optimization (install with pip)
- JSON for storing experimental results
- Ability to run 100+ experimental configurations programmatically

### Codebase Familiarity

- `ska_agent/router/adaptive_router.py` — `PIDController.__init__()` (default gains), `PIDController.update()` (the PID update rule), `ActionScorer.score_action()` (where λ affects routing decisions)
- `ska_agent/core/structures.py` — `PIDConfig` (Kp, Ki, Kd, lambda_max, budget_rate, window_size) and `CostVector` (5 cost dimensions)
- `ska_agent/router/pid_controller.py` — standalone PID implementation

---

## Prep Phase (Weeks 1–3)

### Prep Week 1: Understanding PID Control

**Goal:** Build deep intuition for how the PID controller affects routing behavior.

**Study tasks:**
1. Study PID control basics. A good resource is the Wikipedia article on PID controllers, focusing on: what each term does, the effect of increasing each gain, and common failure modes (oscillation from too-high Kp, integral windup from too-high Ki, noise amplification from too-high Kd).
2. Run the starter code: create a PIDController with default gains, feed it 100 random cost vectors, and plot the lambda trajectory. Observe how lambda responds to cost variations.
3. Experiment with extreme gains: set Kp=10 (very reactive — lambda oscillates wildly), Ki=1 (integral windup — lambda drifts to lambda_max and stays there), Kd=5 (derivative noise — lambda jitters). Plot each trajectory and explain the behavior.
4. Understand the 5 cost dimensions: input_tokens, output_tokens, latency_ms, dollar_cost, meta_overhead. The PID controller maintains a separate lambda for each dimension. Currently all 5 use the same gains — is this appropriate?

**Deliverable:** A notebook showing lambda trajectories under different gain settings, with written explanations of each behavior pattern. Include at least one "pathological" trajectory per gain type.

### Prep Week 2: Evaluation Harness Design

**Goal:** Build the infrastructure to evaluate any PID configuration on a fixed query set.

**Study tasks:**
1. Design the evaluation function: given a PIDConfig, run the router on a fixed set of queries, and return (quality_score, total_cost). Quality can be measured by: number of actions taken (more actions = more thorough), mode diversity (trying different specialists), or answer quality if you have ground truth.
2. Implement the harness with mock specialists (functions that return fixed responses with known costs). This lets you test before the S5 checkpoints arrive.
3. Ensure reproducibility: the same PIDConfig with the same random seed must produce the same score. Test this by running the harness twice and verifying identical results.
4. Profile the harness: how long does one evaluation take? If it takes 5 seconds and you need 100 evaluations, that's 8+ minutes. Plan accordingly.

**Deliverable:** A working `evaluate(config: PIDConfig) -> (quality, cost)` function that is reproducible and profiled for timing.

### Prep Week 3: Search Algorithms

**Goal:** Implement grid search and Bayesian optimization before you have the trained router.

**Study tasks:**
1. Implement grid search over the Kp × Ki × Kd space. Use the starter code grid (6 × 4 × 4 = 96 configurations). Store all results in a structured format (list of tuples: Kp, Ki, Kd, quality, cost).
2. Install scikit-optimize and implement Bayesian optimization using `gp_minimize`. Understand the key parameters: `dimensions` (search bounds), `n_calls` (evaluation budget), and the acquisition function (default is Expected Improvement).
3. Run both on your mock harness. Verify grid search finds the best config exhaustively. Verify BO converges to a similar region with fewer evaluations.
4. Study the Ziegler-Nichols method: set Ki=0, Kd=0, increase Kp until the system oscillates. Record the critical gain K_u and oscillation period T_u. The ZN formulas give: Kp = 0.6·K_u, Ki = 2·Kp/T_u, Kd = Kp·T_u/8.

**Deliverable:** Working grid search and BO implementations, tested on the mock harness. A written description of the Ziegler-Nichols procedure.

---

## Build Phase (Weeks 4–10)

### Build Week 1 (Week 4): Integration with Trained Router

**Tasks:**
- Load the S5-trained router checkpoints. Verify the router produces non-trivial routing decisions (different modes for different queries, multiple actions per query).
- Re-run the evaluation harness with the trained router. Results should be meaningfully different from the mock specialists.
- Re-run grid search on the trained router. Compare the best gains to the defaults (Kp=0.3, Ki=0.01, Kd=0.05). Do the defaults look optimal?

**Expected output:** Grid search completed with trained router. Best gains identified. Comparison table showing default vs best gains and their respective quality/cost.

### Build Week 2 (Week 5): Bayesian Optimization and Comparison

**Tasks:**
- Run BO with 50 evaluations using the trained router.
- Compare BO vs grid search: does BO find a configuration within 5% of grid search best? How many evaluations does it need?
- Plot the BO convergence curve (best score found vs number of evaluations).

**Expected output:** BO results comparable to grid search with fewer evaluations. Convergence curve showing sample efficiency.

### Build Week 3 (Week 6): Ziegler-Nichols Adaptation

**Tasks:**
- Implement the ZN procedure: set Ki=0, Kd=0, sweep Kp from 0.01 to 5.0. For each Kp, run the router and examine the lambda trajectory. Identify the critical Kp where sustained oscillation begins.
- Measure the oscillation period T_u by counting steps between peaks.
- Compute ZN-derived gains and evaluate them.
- Compare ZN gains to grid search and BO results.

**Expected output:** ZN gains derived from oscillation analysis. Lambda trajectory plots showing oscillation detection. ZN gains within the range explored by grid search.

### Build Week 4 (Week 7): Budget Scheduling

**Tasks:**
- Implement three budget schedules: constant (fixed budget throughout), front-loaded (high budget for first 40% of steps, low budget for rest), and cosine annealing (smoothly decreasing from b_max to b_min).
- Run each schedule with the best PID gains on the same query set.
- Plot quality vs total cost for each schedule (a Pareto-like comparison).

**Expected output:** Pareto plot with 3 curves. Front-loaded should outperform constant at the same total cost budget (spending early gathers better context).

### Build Week 5 (Week 8): Full Comparison Matrix

**Tasks:**
- Create a comprehensive comparison: 4 gain methods (default, grid best, BO best, ZN) × 2 schedules (constant, front-loaded) = 8+ configurations.
- Run all on the same query set. Build a results table with columns for: gains, schedule, quality score, total cost, number of actions, and lambda stability (variance of lambda trajectory).
- Identify the overall winner.

**Expected output:** An 8+ row comparison table. Clear winner identified with explanation of why it wins.

### Build Week 6 (Week 9): Sensitivity and Robustness

**Tasks:**
- Test the winning configuration on different query distributions (more COMPUTE queries, more MULTI_STEP queries). Does it remain optimal?
- Test sensitivity to budget levels: does the winning config work well at 0.5× and 2× the default budget?
- Run the winning config with 3 random seeds to measure variability.

**Expected output:** Robustness analysis showing whether the recommendation generalizes. Sensitivity to budget level documented.

### Build Week 7 (Week 10): Written Analysis

**Tasks:**
- Write a 2–3 page analysis covering: the PID controller's role, all tuning methods and their results, budget scheduling comparison, and the recommended configuration.
- Include: lambda trajectory plots for the best and worst configurations, Pareto frontier, convergence curves for BO, and the comparison table.
- The recommendation should be concrete: "Use Kp=X, Ki=Y, Kd=Z with front-loaded scheduling (budget_rate=A for the first 40% of queries, budget_rate=B thereafter)."

**Expected output:** Written report with data-backed recommendation.

---

## What "Done" Looks Like

1. An evaluation harness that measures cost-quality tradeoffs for any PID configuration
2. Three tuning methods implemented and compared (grid search, Bayesian optimization, Ziegler-Nichols)
3. Three budget scheduling policies evaluated (constant, front-loaded, cosine)
4. A comprehensive comparison table with 8+ configurations
5. Written analysis with a specific, justified recommendation for default PID gains and budget schedule

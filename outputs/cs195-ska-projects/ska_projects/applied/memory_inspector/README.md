# A2: Spectral Memory Inspector + Anomaly Detection — Full Project Guide

## Project Classification: Coding/Visualization with Light Math

**Tier:** Starter | **GPU:** None | **Team Size:** 1 | **Total Duration:** 10 weeks (3 prep + 7 build)

**Dependencies:** None. Fully self-contained.

---

## Project Summary

You are building a visual debugger for SharedSpectralMemory — the communication backbone between agents. The first half builds a dashboard that shows operator spectra, condition number evolution, and read/write activity. The second half adds automated anomaly detection that identifies operator degradation (condition number spikes, spectral collapse) and triggers recovery actions (rebuild, ridge boost, selective forget).

---

## Motivation: Why This Matters for SKA-Agent

The shared spectral memory is the defining architectural innovation that separates SKA-Agent from conventional multi-agent systems. In a standard pipeline, Agent A finishes its work and passes 2,000 tokens of findings into Agent B's context window — growing B's KV cache by ~32MB and consuming precious context space. In SKA-Agent, Agent A writes key vectors that update a rank-64 Koopman operator (~32KB), and Agent B queries that operator through the power spectral filter to retrieve what A learned. The operator captures temporal dynamics (how A's findings evolved across steps), amplifies consistent signals, and suppresses noise — all in a fixed-size matrix that doesn't grow with the amount of information written.

But operators can go wrong. When agents write keys from very different distributions (a retriever finding financial data and a code executor producing calculation results), the Gram matrix can become ill-conditioned — its condition number κ explodes, the Cholesky factorization amplifies numerical errors, and reads return garbage. When one direction dominates (spectral collapse), the operator loses its ability to filter — everything looks the same through the power filter. These failures are silent: the system doesn't crash, it just produces subtly wrong results.

Your inspector makes these failure modes visible, and your anomaly detector makes them recoverable. In a live multi-agent workflow where the retriever, coordinator, and code executor are all writing to different slots of the multi-head Koopman module, automated health monitoring is essential. The HealthMonitor you build detects condition number spikes (an agent's key distribution shifted suddenly) and spectral collapse (the operator has degenerated), then triggers recovery — rebuilding from recent keys, boosting the ridge regularization, or selectively forgetting old state. This is the operational safety layer that keeps the spectral memory protocol reliable under real workloads.

---

## Starting Requirements

### Conceptual Prerequisites (Light Math)

You do NOT need deep linear algebra for this project, but you do need to understand a few concepts at an intuitive level:

- **Shared memory as a matrix:** The shared memory stores a 64×64 matrix (the Koopman operator). Agents write to it by providing key vectors. Agents read from it by querying with a vector and getting a transformed vector back. Think of it as a shared whiteboard that compresses many notes into one small matrix.
- **Condition number (κ):** A health metric. κ ≈ 1 means healthy. κ > 10,000 means the matrix is "sick" and reads may return garbage. You display this number — you don't need to compute it from scratch (the codebase does that).
- **Singular values:** The matrix has 64 singular values that describe its "strength" in different directions. A healthy matrix has a few large singular values (strong directions) and many small ones (filtered out). A degenerate matrix has all similar values (no filtering). You display these as a bar chart.
- **Spectral radius:** The largest singular value. If it's above 1, the operator amplifies; below 1, it contracts. You display this number.

### Technical Prerequisites (Coding-Focused)

- **Python and Streamlit:** Same as A1. You build a web dashboard with interactive charts.
- **Plotly:** Bar charts (singular values), line charts (condition number over time), scatter plots.
- **NumPy basics:** Creating arrays, matrix operations (`np.outer`, `np.linalg.svd`, `np.linalg.norm`). You'll call these functions but don't need to understand the algorithms.
- **Basic threshold-based detection:** If metric > threshold, flag an anomaly. This is the core of the HealthMonitor.

### Codebase Familiarity

- `ska_agent/shared_memory/spectral_memory.py` — `SharedSpectralMemory` with `write()`, `read()`, `operator` property, `reset()`, `rebuild_from_scratch()`. You call these methods and display the results.
- `ska_agent/core/structures.py` — `SharedOperator` with `A_w` (the matrix), `condition_number`, `num_tokens_seen`.

---

## Prep Phase (Weeks 1–3)

### Prep Week 1: Streamlit, Plotly, and the Memory API

**Goal:** Get a working app that creates a SharedSpectralMemory and displays basic metrics.

**Tasks:**
1. Install Streamlit and Plotly. Run the starter code for the minimal spectrum viewer.
2. Understand the SharedSpectralMemory API: `write(keys, source_agent)` adds keys to the memory. The `operator` property returns a `SharedOperator` object with the matrix `A_w` and health metrics.
3. Create a simple app: initialize memory, write random keys, display condition number and spectral radius as `st.metric()` widgets.
4. Add a slider controlling the number of writes. Observe how metrics change with more writes.
5. Compute singular values with `np.linalg.svd(op.A_w, compute_uv=False)` and display as a bar chart.

**Deliverable:** A working Streamlit app showing singular value bar chart, condition number, and spectral radius, updating as you move the writes slider.

### Prep Week 2: VIM/CLI Basics and Timeline Visualization

**Goal:** Add temporal tracking and the read test panel.

**Tasks:**
1. **VIM practice:** Edit your dashboard.py in VIM. Add a new section, modify chart parameters, save and let Streamlit auto-reload.
2. Build the condition number timeline: create a fresh memory, write keys one batch at a time, record κ after each write, display as a line chart. Add a red dashed line at the alert threshold (10^4).
3. Build a write activity log: a table showing each write event with columns for write number, source agent, key count, and post-write κ.
4. Build the read test panel: button that sends a random query through the operator at K=1,2,3,5 (different power iteration depths) and displays output norms.

**Deliverable:** Dashboard with timeline, activity log, and read test panel all working.

### Prep Week 3: Multi-Agent Simulation and Anomaly Concepts

**Goal:** Build the simulation mode and understand what anomalies look like.

**Tasks:**
1. Simulate 3 agents writing keys from different distributions: agent_0 writes from N(0, 0.1·I), agent_1 writes from N(μ₁, 0.1·I), agent_2 writes from N(μ₂, 0.1·I) where μ₁ and μ₂ are different random directions. Observe how the spectrum and condition number depend on agent diversity.
2. Create an anomaly scenario: agent_0 writes normal keys for 50 steps, then suddenly shifts to a very different distribution. Observe the condition number spike. This is the pattern your anomaly detector will need to catch.
3. Study the HealthMonitor skeleton from the starter code. Understand spike detection (κ jumped by more than threshold×) and spectral collapse (ratio of first to second singular value is extreme).
4. Plan the recovery policies: rebuild from scratch (drastic but guaranteed), ridge boost (increase regularization temporarily), selective forget (discard old keys, keep recent).

**Deliverable:** Multi-agent simulation showing agent diversity effects. At least one anomaly scenario where κ spikes visibly. Written plan for the anomaly detector.

---

## Build Phase (Weeks 4–10)

### Build Week 1 (Week 4): Complete Visual Inspector

**Tasks:**
- Polish the dashboard layout: spectrum viewer, κ timeline, write log, and read test in a clean layout.
- Add the multi-agent simulation with agent selection controls (which agents are active, how many writes each).
- Add controls for memory parameters: rank, ridge_eps, power_K.

**Expected output:** A complete visual inspector showing operator health across different scenarios.

### Build Week 2 (Week 5): HealthMonitor — Spike and Collapse Detection

**Tasks:**
- Implement the HealthMonitor class with spike detection (κ jumped by >10× in one write) and spectral collapse detection (σ₁/σ₂ > 100).
- Wire it into the dashboard: anomalies appear as red markers on the condition number timeline.
- Generate a synthetic workload with 2 injected anomaly events. Verify detection within 1–2 writes of the anomaly.
- Measure false positive rate on normal workloads (200 writes with no distribution shift). Target: <5% false positives.

**Expected output:** HealthMonitor detecting injected anomalies with low false positive rate. Red markers on the timeline.

### Build Week 3 (Week 6): Recovery Policies

**Tasks:**
- Implement 3 recovery policies: rebuild (clear all state, rebuild from recent keys), ridge boost (temporarily increase ridge_eps by 10×), selective forget (discard oldest 50% of keys, rebuild from the rest).
- Wire each to a button in the dashboard. When an anomaly is detected, the user can click to apply a recovery.
- Verify each policy brings κ below the alert threshold after recovery.

**Expected output:** Three working recovery policies, each successfully reducing κ after an anomaly.

### Build Week 4 (Week 7): Policy Comparison Experiment

**Tasks:**
- Design a standardized test: 200 writes with 3 injected anomaly events (at write 60, 120, 180).
- Run the test with each recovery policy. Measure: κ trajectory, number of recovery events triggered, max κ reached, and read output quality (norm of read output before vs after recovery).
- Build a comparison table and comparison plots.

**Expected output:** Comparison table showing all 3 policies on 4+ metrics. At least one policy is clearly better on each metric.

### Build Week 5 (Week 8): Automated Recovery

**Tasks:**
- Make recovery automatic: when the HealthMonitor detects an anomaly, it triggers the best recovery policy without user intervention.
- Add configurable thresholds: the user can set spike sensitivity and collapse sensitivity.
- Add an anomaly event log showing timestamps, anomaly types, and recovery actions taken.

**Expected output:** Automatic anomaly detection and recovery. Configurable thresholds. Event log.

### Build Week 6 (Week 9): Stress Testing and Edge Cases

**Tasks:**
- Stress test: 1000 writes with 10 injected anomalies. Does automatic recovery keep the system healthy?
- Edge cases: what happens when all agents write the same keys (total redundancy)? When one agent writes keys with very large norms? When ridge_eps is set to 0?
- Tune the detection thresholds and recovery parameters for best overall behavior.

**Expected output:** Stress test results showing the system handles sustained anomalies. Edge cases documented.

### Build Week 7 (Week 10): Integration and Polish

**Tasks:**
- Integrate HealthMonitor into SharedSpectralMemory as an optional component: `SharedSpectralMemory(rank=64, health_monitor=True)`.
- Polish the dashboard: clean layout, clear labels, intuitive controls.
- Write a brief description visible in the dashboard explaining what each panel shows (for someone unfamiliar with SKA).

**Expected output:** Clean, integrated HealthMonitor. Dashboard is screenshot-ready and self-explanatory.

---

## What "Done" Looks Like

1. A visual inspector showing: operator spectrum, condition number timeline, write activity log, and read test panel
2. A HealthMonitor detecting spikes and spectral collapse with <5% false positive rate
3. Three recovery policies (rebuild, ridge boost, selective forget) implemented and compared
4. A quantitative comparison experiment showing which policy performs best under what conditions
5. Automatic anomaly detection and recovery integrated into the dashboard
6. HealthMonitor available as an optional component of SharedSpectralMemory

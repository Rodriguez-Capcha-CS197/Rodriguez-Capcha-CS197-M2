# A1: Router Decision Dashboard + Comparative Analysis — Full Project Guide

## Project Classification: Coding/Application-Heavy (Starter)

**Tier:** Starter | **GPU:** None | **Team Size:** 1–2 | **Total Duration:** 10 weeks (3 prep + 7 build)

**Dependencies:** Requires trained router checkpoints from S5 Router Training (delivered at S5's build week 3). Build the dashboard UI in prep and early build weeks with mock data, then plug in the real router.

---

## Project Summary

You are building a web dashboard that visualizes the AdaptiveRouter's decision process — mode selection, action scoring, PID cost dynamics — and then using that dashboard to conduct a systematic analysis of routing behavior, identifying where the router succeeds and where it fails.

The first half is engineering (build the dashboard). The second half is analysis (use the dashboard to study routing behavior and produce a written report).

---

## Motivation: Why This Matters for SKA-Agent

SKA-Agent's router doesn't make a single yes/no decision — it makes a structured sequence of decisions: classify the query into a collaboration mode, select a DAG template, then at each DAG node evaluate candidate specialist actions using a scoring function that balances predicted quality against a PID-controlled cost penalty across five dimensions. This produces rich, multi-step execution traces — but right now, the only way to see what the router decided is `verbose=True` print statements scrolling past in a terminal.

The dashboard you build makes the router's decisions visible, and once visible, analyzable. This matters because the router is the central coordinator of the entire multi-agent system. When it classifies a COMPUTE query as LOOKUP, the code executor never runs and the user gets a text answer instead of a calculation. When the PID controller's lambda vector is too aggressive, the router stops after one action even when chaining two specialists would produce a better answer. When the RewardPredictor undervalues the retriever for multi-document queries, the system falls back to the reasoner alone and misses cross-document evidence.

Your systematic analysis — mode confusion patterns, cost-quality Pareto frontiers, failure pattern identification — provides the diagnostic evidence that other teams need to improve their components. The M4 team needs to know which mode pairs get confused most often. The S4 team needs to see how PID budget settings affect routing behavior. The S5 team needs confusion matrices to target their training data. Your dashboard and analysis become the feedback loop that the whole project ecosystem relies on.

---

## Starting Requirements

### Technical Prerequisites (Coding-Focused)

- **Python basics:** Functions, dictionaries, lists, loops, file I/O. You do NOT need deep ML knowledge for this project.
- **Streamlit:** A Python framework for building web dashboards. You create UI elements (text inputs, buttons, charts, columns) by calling Python functions. Streamlit re-runs your script top-to-bottom on every interaction. You need to understand this execution model.
- **Plotly:** For interactive charts. You'll make bar charts (mode probabilities), line charts (PID dynamics), scatter plots (confidence vs actions), and heatmaps (cost analysis).
- **Session state:** Streamlit's `st.session_state` persists data across interactions within a session. You need this for tracking query history and PID dynamics.
- **JSON:** For saving and loading batch evaluation results.
- **No math required.** You use the router as a black box — you call it and visualize its outputs. You don't need to understand the internal math.

### Tools to Install

- `pip install streamlit plotly` — the dashboard framework and charting library
- `pip install graphviz` — for DAG visualization (and install the graphviz system package)
- sentence-transformers (the QueryEncoder uses it internally)

### Understanding the Router (Conceptual, Not Mathematical)

You need to understand the router's behavior at a high level:
1. It classifies queries into one of 4 modes (LOOKUP, MULTI_DOC, COMPUTE, MULTI_STEP)
2. For each step in the mode's workflow, it scores candidate actions (specialist + target combinations)
3. It executes the highest-scoring action if the score is positive, otherwise stops
4. A PID controller adjusts cost penalties over time, making the router more or less conservative

You do NOT need to understand how mode classification works internally, how scores are computed, or the PID math. You just display the outputs.

---

## Prep Phase (Weeks 1–3)

### Prep Week 1: Streamlit and Plotly Basics

**Goal:** Get a working Streamlit app that displays interactive charts.

**Tasks:**
1. Install Streamlit. Run the starter code: `streamlit run dashboard.py`. Make sure it opens in your browser.
2. Learn the Streamlit layout model: `st.columns()` for side-by-side elements, `st.expander()` for collapsible sections, `st.sidebar` for navigation, `st.metric()` for KPI display.
3. Build a practice app: text input, button, and a Plotly bar chart that updates when you click the button. This mirrors the core dashboard interaction.
4. Learn `st.session_state`: create a counter that persists across button clicks. Then create a list that accumulates items across interactions.
5. Learn Plotly basics: `go.Bar()`, `go.Scatter()`, `go.Figure()`, `fig.update_layout()`. Create one bar chart and one line chart with custom colors and labels.

**Deliverable:** A working Streamlit app with a text input, a button, a Plotly bar chart, and a line chart that updates from session state.

### Prep Week 2: VIM/CLI and Dashboard Layout

**Goal:** Set up your development workflow and build the dashboard skeleton with mock data.

**Tasks:**
1. **VIM basics:** Open `dashboard.py`, navigate to functions, edit parameters, save and re-run. Streamlit auto-reloads on file save, so the workflow is: edit in VIM → save → browser refreshes.
2. **Command line workflow:** Keep a terminal running `streamlit run dashboard.py` and another for editing. Practice switching between them.
3. Build the dashboard layout with mock data:
   - Mode selection panel: horizontal bar chart with 4 bars (LOOKUP, MULTI_DOC, COMPUTE, MULTI_STEP)
   - Action candidates table: list of (specialist, target, score) rows
   - Execution trace: expandable sections showing each routing step
   - PID dynamics: line chart with 5 dimensions over time
4. Wire the QueryEncoder (pretrained MiniLM — no dependency on S5): `from ska_agent.router.adaptive_router import QueryEncoder; encoder = QueryEncoder("all-MiniLM-L6-v2")`. Display the real 384-dim embedding for typed queries.

**Deliverable:** Dashboard skeleton showing all 4 panels with mock data. QueryEncoder producing real embeddings.

### Prep Week 3: DAG Visualization and History

**Goal:** Complete the mock-data version of the dashboard.

**Tasks:**
1. Implement DAG visualization using `st.graphviz_chart()`. Display the MODE_TEMPLATES for all 4 modes. Color nodes green if "executed" (mock for now).
2. Implement query history using session state: every query typed is logged with timestamp, selected mode, and action count. Display as a table.
3. Implement PID dynamics tracking: store lambda vectors in session state across queries. Display as a multi-line chart.
4. Test the full dashboard flow: type a query, see mode probabilities, see action candidates, see the DAG, see PID dynamics update. Everything uses mock data but the interactions work.

**Deliverable:** A complete dashboard with all UI components working on mock data. Ready for real router integration.

---

## Build Phase (Weeks 4–10)

### Build Week 1 (Week 4): Real Router Integration

**Tasks:**
- Load S5 checkpoints with `load_trained_router()`. Replace all mock data with real router outputs.
- Display real mode probabilities, real action scores, real execution traces.
- Add batch evaluation mode: load a JSON file of queries, run all of them through the router, save traces.

**Expected output:** Dashboard showing real router decisions. At least some queries result in 2+ actions. Batch mode processes 50 queries and saves results.

### Build Week 2 (Week 5): Mode Confusion Analysis

**Tasks:**
- Run 100+ queries through the router. For each, record the max mode probability (confidence).
- Build a scatter plot: confidence vs number of actions taken. Is there a correlation?
- Identify queries where confidence < 0.5 (the model is uncertain). Examine these by hand — why is the model confused? Are these genuinely ambiguous queries or model failures?

**Expected output:** Scatter plot showing confidence vs action count. At least 3 specific "confused" queries identified and explained.

### Build Week 3 (Week 6): Cost-Quality Pareto Analysis

**Tasks:**
- Run the same 50 queries at 5 different PID budget_rate settings (0.5×, 0.75×, 1×, 1.5×, 2× of default).
- For each setting, measure: total cost (sum of all cost vectors) and routing "quality" (number of actions taken, mode diversity, or another proxy).
- Plot the Pareto frontier: total cost vs quality.

**Expected output:** Pareto plot with 5 points. A clear knee point where tightening the budget starts significantly reducing quality.

### Build Week 4 (Week 7): Failure Pattern Identification

**Tasks:**
- Group queries by mode. Compute per-mode statistics: average confidence, average action count, fraction of queries where the router stops after 0 actions (a failure — no specialist was invoked).
- Find query patterns where the router consistently picks the wrong mode or stops too early.
- Document at least 3 concrete failure patterns with specific example queries for each.

**Expected output:** Per-mode statistics table. 3+ failure patterns documented with examples and explanations.

### Build Week 5 (Week 8): Dashboard Polish and Advanced Features

**Tasks:**
- Add a comparison view: run the same query with different settings side by side.
- Add filtering and sorting to the batch results view.
- Improve visual design: colors, layout, spacing, labels.
- Add download buttons for batch results and charts.

**Expected output:** A polished dashboard that someone unfamiliar with SKA could understand within 30 seconds.

### Build Week 6 (Week 9): Extended Analysis

**Tasks:**
- Run a larger batch evaluation (200+ queries) covering diverse query types.
- Analyze whether the router's behavior changes across different document types.
- Identify the single most impactful improvement the router could make (e.g., "the model confuses LOOKUP and MULTI_DOC 40% of the time — fixing this would improve routing for 25% of queries").

**Expected output:** Extended analysis on 200+ queries. Most impactful improvement identified.

### Build Week 7 (Week 10): Written Report

**Tasks:**
- Write a 2–3 page report documenting findings from weeks 5–9.
- Structure: Introduction (what the router does), Dashboard (what it shows), Findings (confusion analysis, Pareto analysis, failure patterns), Recommendations (3+ actionable improvements).
- Include key charts: mode probability distributions, confidence scatter plots, Pareto frontier, failure examples.

**Expected output:** Written report with 3+ data-backed recommendations for improving the router.

---

## What "Done" Looks Like

1. A working Streamlit dashboard with: mode probability charts, action score tables, execution trace display, PID dynamics tracking, and DAG visualization
2. Batch evaluation mode that processes 50+ queries and exports traces
3. Mode confusion analysis identifying uncertain queries
4. Cost-quality Pareto analysis across budget settings
5. At least 3 concrete routing failure patterns documented
6. A written report with actionable recommendations for router improvement

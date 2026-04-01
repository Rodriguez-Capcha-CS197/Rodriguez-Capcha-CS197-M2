# S2: Streaming Bounded-Memory Segmentation — Full Project Guide

## Project Classification: Algorithms/Math-Hybrid

**Tier:** Intermediate | **GPU:** Minimal (one-time embedding) | **Team Size:** 1–2 | **Total Duration:** 10 weeks (3 prep + 7 build)

**Dependencies:** None

---

## Project Summary

You are building a streaming version of the GeometryLearner's dynamic programming segmentation algorithm. The batch version loads all sentences into memory, computes all embeddings, then runs DP over the full corpus. Your streaming version processes sentences one at a time with O(lookback_k) memory, produces identical segments to the batch algorithm, and emits finalized segments incrementally without waiting for the entire document.

---

## Motivation: Why This Matters for SKA-Agent

The geometry learner (Stage I of the SKA pipeline) is the foundation of everything that follows. It takes raw corpus text and segments it into semantically coherent units using dynamic programming — finding natural topic boundaries instead of chopping text into arbitrary 512-token chunks. These segments become the atomic units for retrieval: the PricingEngine selects segments, segment embeddings are used to build Koopman operators, and the quality of the operator depends directly on whether the segments represent coherent topics.

The batch algorithm works well but has a scaling problem: it loads all sentences, embeds them all, then runs DP over the full corpus. For a large document collection (the system was designed for 89,000 pages of Treasury Bulletins), this means tens of gigabytes of embeddings in memory before segmentation even starts. The DP algorithm itself has a lookback_k parameter (default 50) that already bounds how far back it looks — dp[i] only depends on dp[i-50] through dp[i-1]. This bounded dependency structure means the algorithm is inherently local, even though the batch implementation treats it as global.

Your streaming version exploits this locality. By maintaining only a sliding window of O(lookback_k) DP values, prefix sums, and parent pointers, it processes sentences one at a time with constant memory and emits finalized segments incrementally. This enables two things that matter for the full system: first, the offline pipeline can process arbitrarily large corpora without running out of memory; second, in a live deployment, new documents can be ingested and segmented on-the-fly, with finalized segments immediately available for retrieval and operator construction, rather than waiting for the entire document to be processed.

---

## Starting Requirements

### Mathematical/Algorithmic Prerequisites

- **Dynamic programming (DP):** You must understand DP well — not just how to use it, but how to derive recurrences, prove optimality, and trace back solutions. The segmentation DP is a 1D optimization over boundaries.
- **The DP recurrence:** dp[0] = 0; dp[i] = min over j in [max(0, i-k), i) of {dp[j] + cost(j,i) + λ}. This finds the minimum-cost way to partition sentences [0, i) into segments, where λ penalizes each additional segment. You need to understand what each term does and why the lookback_k bound makes it a sliding-window DP.
- **Prefix sums for cost computation:** The internal cost of a segment [j, i) is the sum of consecutive cosine distances between sentence embeddings within that segment. Prefix sums allow O(1) cost queries: cost(j, i) = prefix[i] - prefix[j]. You need to understand this technique.
- **Segment finalization insight:** A segment [a, b) is finalized when no future decision can change it. Since the DP only looks back k positions, once we've computed dp[i] for all i ≥ b + k, the boundary at b is locked. This means segments can be emitted with a delay of k positions.
- **Circular buffers:** The streaming algorithm maintains only the most recent k values of dp, prefix sums, and parent pointers. A circular buffer wraps around when it reaches capacity, overwriting the oldest entries. You need to understand the modular arithmetic for indexing.
- **Cosine distance:** distance(u, v) = 1 - (u·v)/(||u||·||v||). This is computed between consecutive sentence embeddings to measure topic change.

### Programming Prerequisites

- Python with NumPy for embedding operations and distance computations
- Understanding of memory profiling (tracemalloc) and wall-clock benchmarking
- Ability to write precise correctness tests (comparing streaming output against a known-correct batch algorithm, element by element)
- Comfort with modular arithmetic for circular buffer indexing

### Codebase Familiarity

- `ska_agent/core/geometry.py` — `GeometryLearner.learn_geometry()`, the batch DP you are replacing. Read this carefully and trace through it by hand.
- `ska_agent/utils/math_utils.py` — `MathUtils.compute_prefix_sums()`, `compute_pairwise_distances()`, `segment_internal_cost()`. These are the batch helpers you'll replace with streaming equivalents.
- `ska_agent/pipeline.py` — `OfflinePipeline.process()`, the full batch pipeline you'll eventually add a streaming mode to.

---

## Prep Phase (Weeks 1–3)

### Prep Week 1: Understanding the Batch DP

**Goal:** Understand the batch segmentation algorithm deeply enough to derive its streaming equivalent.

**Study tasks:**
1. Read `GeometryLearner.learn_geometry()` line by line. Write down the DP recurrence in mathematical notation. Identify every array that is allocated and its size.
2. Hand-trace the DP on a small example: 20 sentences with synthetic embeddings. Compute cosine distances between consecutive sentences, build prefix sums, run the DP, and extract segments. Do this in a notebook, printing intermediate values.
3. Understand the lookback_k parameter: the DP only considers split points within the last k positions. This means dp[i] depends only on dp[i-k] through dp[i-1], not on any earlier values. This bounded dependency is what enables streaming.
4. Understand the backtracking step: after computing all dp values, the batch algorithm traces back through parent pointers to extract segment boundaries. In the streaming version, you'll need to emit segments as they become finalized, not wait until the end.

**Deliverable:** A hand-traced example of the batch DP on 20 sentences, showing dp values, parent pointers, and final segments. A written explanation of why the lookback bound enables streaming.

### Prep Week 2: Streaming DP Design

**Goal:** Design the streaming algorithm on paper.

**Study tasks:**
1. Design the streaming state: at any point, you only need: the last k values of dp, the last k parent pointers, the last k prefix sum values, and the last k sentence embeddings (for computing distances to new sentences). All of these fit in circular buffers of size k.
2. Work through the finalization logic: after computing dp[i], which segments are guaranteed to be final? A segment ending at position b is final when i ≥ b + k, because no future dp value can choose a split point before b. This means you can emit segments with a delay of k positions.
3. Handle the edge case: when the stream ends (flush), you need to backtrack from the final dp value to extract any remaining segments that weren't yet finalized.
4. Analyze memory: the streaming algorithm uses O(k) arrays plus whatever embedding memory is needed. For k=50 and d=384, this is a few KB regardless of document size.

**Deliverable:** Pseudocode for the streaming DP including the `feed()` and `flush()` methods. A diagram showing which segments are finalized at each step.

### Prep Week 3: Circular Buffer and Streaming Prefix Sums

**Goal:** Implement the data structures needed for the streaming DP.

**Study tasks:**
1. Implement the CircularBuffer class (starter code provided). Test it thoroughly: append past capacity, verify wrap-around, test edge cases (empty buffer, single element, exactly at capacity).
2. Implement streaming prefix sum computation: when a new embedding arrives, compute the cosine distance to the previous embedding and add it to the running prefix sum. This replaces the batch `compute_pairwise_distances()` and `compute_prefix_sums()`.
3. Verify your streaming prefix sums match the batch computation: compute distances and prefix sums both ways for 100 sentences and check they agree.
4. Implement the streaming cost query: cost(j, i) = prefix[i] - prefix[j], using the circular buffer to retrieve the correct values.

**Deliverable:** Working CircularBuffer and streaming prefix sum computation, verified against batch computation on 100 sentences.

---

## Build Phase (Weeks 4–10)

### Build Week 1 (Week 4): Core Streaming DP Step

**Tasks:**
- Implement `feed()`: compute the cosine distance to the previous embedding, update the prefix sum buffer, compute dp[pos] using the lookback window, and store the parent pointer.
- Do NOT implement finalization yet — just get the dp values right.
- Verify: run both batch and streaming on the same 100 sentences, compare dp values at every position. They must match exactly.

**Expected output:** dp values from streaming match batch dp values for 100 sentences. Print both side by side to verify.

### Build Week 2 (Week 5): Segment Finalization

**Tasks:**
- Implement the finalization logic in `feed()`: when a segment is guaranteed to be final (its end boundary is more than k positions behind the current position), extract it and return it.
- Implement `flush()`: after the stream ends, backtrack to extract remaining segments.
- Run `test_streaming_matches_batch()` on N=500 with lambda=0.3. Segments must match exactly (same start_idx, same end_idx, same number of segments).

**Expected output:** The correctness test passes. Streaming produces identical segments to batch.

### Build Week 3 (Week 6): Edge Cases and Robustness

**Tasks:**
- Test edge cases: empty input, single sentence, all-identical embeddings (all distances are 0), very high lambda (one giant segment), very low lambda (each sentence is its own segment), min/max segment size constraints.
- Run correctness tests on 10 different random seeds and 3 different lambda values (0.1, 0.3, 0.5).
- Fix any bugs found.

**Expected output:** All edge cases handled correctly. Correctness verified across multiple seeds and lambda values.

### Build Week 4 (Week 7): StreamingOfflinePipeline

**Tasks:**
- Build a `StreamingOfflinePipeline` that reads text line by line, embeds in micro-batches of 32 (for efficiency), and feeds into StreamingGeometryLearner.
- Process a real document (10+ pages, from a PDF). Compare output against the batch OfflinePipeline, segment for segment.

**Expected output:** Full pipeline runs end-to-end on a real document. Segments match batch output exactly.

### Build Week 5 (Week 8): Memory Benchmarks

**Tasks:**
- Run streaming segmentation on N=1K, 10K, 100K synthetic sentences. Measure peak memory using tracemalloc.
- Verify that peak memory is constant (within 10%) across all three corpus sizes. The streaming algorithm should use O(k) memory regardless of N.
- Compare against batch memory usage (which grows linearly with N).

**Expected output:** Memory benchmark table showing constant peak memory for streaming vs linear growth for batch. Clear documentation of the exact memory footprint.

### Build Week 6 (Week 9): Wall-Clock Benchmarks

**Tasks:**
- Compare streaming vs batch wall-clock time on N=1K, 10K, 100K. Include both embedding time (the same for both) and segmentation time.
- The streaming segmentation may be slower than batch due to Python loop overhead (batch uses vectorized NumPy operations). Document the overhead factor.
- Identify the crossover point: at what N is the memory saving worth the time overhead?

**Expected output:** Benchmark table with wall-clock times. Streaming segmentation time within 2× of batch. Memory savings clearly documented.

### Build Week 7 (Week 10): Integration and Report

**Tasks:**
- Add `StreamingGeometryLearner` to `core/geometry.py`. Update `OfflinePipeline` with a `streaming=True` option.
- Run integration test: `OfflinePipeline(streaming=True)` produces identical segments to `OfflinePipeline(streaming=False)` on a real PDF.
- Write a 2-page report covering: the streaming DP algorithm, finalization logic, correctness guarantees, memory and time benchmarks, and recommended use cases.

**Expected output:** Clean integration with a config flag. Written report with algorithm description and benchmark results.

---

## What "Done" Looks Like

1. A streaming DP segmentation algorithm that produces identical results to the batch algorithm
2. Correct segment finalization with lookback_k delay
3. All edge cases handled and tested across multiple seeds and lambda values
4. Proven constant-memory operation (O(lookback_k) regardless of document size)
5. Wall-clock benchmarks comparing streaming vs batch
6. Integration into the codebase with a streaming mode flag

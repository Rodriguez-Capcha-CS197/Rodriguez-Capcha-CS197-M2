# A3: OfficeQA Demo App + Retrieval Strategy Comparison — Full Project Guide

## Project Classification: Coding/Application-Heavy (Starter)

**Tier:** Starter | **GPU:** Minimal (CPU embedder) | **Team Size:** 1–2 | **Total Duration:** 10 weeks (3 prep + 7 build)

**Dependencies:** None. Fully self-contained.

---

## Project Summary

You are building a web app where users upload a PDF, watch it get segmented into topically coherent chunks, ask questions, and see retrieved segments with answers. In the second half, you implement 3 retrieval baselines (top-k cosine, fixed-chunk, BM25) and run a systematic comparison against SKA's pricing-guided retrieval.

---

## Motivation: Why This Matters for SKA-Agent

SKA-Agent's retrieval pipeline has two innovations that standard RAG systems don't: geometry-learned segmentation and pricing-guided selection. Standard RAG chops documents into fixed-size chunks (every 512 tokens), embeds them, and returns the top-k by cosine similarity. This has well-known problems: relevant information gets split across chunk boundaries, irrelevant padding dilutes the embedding, and top-k retrieval has no mechanism to avoid redundancy (you can get 5 segments that all say the same thing).

SKA-Agent's approach is different at both stages. The geometry learner uses dynamic programming to find natural topic boundaries — segments adapt to the document's structure rather than imposing an arbitrary grid. Then the PricingEngine selects segments using an optimization framework: each segment's inclusion is justified by its information gain (measured via the Schur complement of the query-segment projection) minus a cost penalty (λ for sparsity, η for redundancy). A segment is only included if it provides genuinely new information that isn't already covered by previously selected segments. This is the retrieval mechanism that feeds context to the Koopman operator construction — and the quality of that context directly determines the quality of the spectral representation that the multi-agent system communicates through.

Your demo app makes this pipeline tangible (a user can upload a PDF and see it work), and your comparison study validates it empirically. By measuring pricing-guided retrieval against top-k cosine, fixed-chunk, and BM25 baselines on real documents, you provide the first concrete evidence of whether the mathematical machinery actually produces better retrieval in practice. If it does, you've justified a core architectural choice. If it doesn't for certain query types, you've identified where the system needs improvement — which is equally valuable.

---

## Starting Requirements

### Conceptual Prerequisites (Light Math/Retrieval)

- **Embeddings:** Text is converted to a vector (array of 384 numbers) by a pretrained model. Similar texts have similar vectors (high cosine similarity). You don't need to understand how the embedding model works — just that it converts text → vector.
- **Cosine similarity:** sim(u, v) = (u · v) / (||u|| · ||v||). Ranges from -1 to 1. Higher means more similar. This is how retrieval systems rank candidates.
- **Retrieval basics:** Given a query, find the most relevant text segments. Top-k cosine retrieval: embed the query, embed all segments, return the k segments with highest cosine similarity. This is the standard RAG approach.
- **Dynamic segmentation vs fixed chunking:** Fixed chunking splits text every 512 tokens regardless of content. Dynamic segmentation (the SKA approach) uses a DP algorithm to find natural topic boundaries. You'll compare both.
- **Pricing-guided retrieval:** SKA's approach adds segments one at a time, only including a segment if its "information gain" exceeds a cost threshold (lambda). This prevents retrieving redundant segments. You display the results — you don't need to understand the pricing math.
- **BM25:** A keyword-based retrieval method. Unlike cosine similarity (which works on embeddings), BM25 counts word overlaps with TF-IDF weighting. Good for exact keyword matches, weaker for semantic similarity.

### Technical Prerequisites (Coding-Focused)

- **Python and Streamlit:** Web dashboard framework.
- **File handling:** Uploading files in Streamlit, using temporary files, reading PDFs.
- **Plotly:** For segment statistics charts and comparison visualizations.
- **pip install:** You'll need several packages: streamlit, plotly, sentence-transformers, rank_bm25.

### Codebase Familiarity

- `ska_agent/pipeline.py` — `OfflinePipeline.process()` takes raw text and returns segments.
- `ska_agent/core/pricing.py` — `PricingEngine.retrieve()` takes a query and returns selected segments.
- `ska_agent/evaluation/officeqa.py` — `DocumentProcessor.process_pdf()` extracts text from PDFs.
- `ska_agent/models/embedding.py` — `Embedder` for encoding text.

---

## Prep Phase (Weeks 1–3)

### Prep Week 1: Streamlit and PDF Processing

**Goal:** Get a working app that accepts a PDF upload and displays extracted text.

**Tasks:**
1. Install all dependencies: `pip install streamlit plotly sentence-transformers rank_bm25`.
2. Run the starter code for PDF upload and text extraction. Upload a real PDF (any public document — a government report, academic paper, etc.).
3. Display the extracted text and table count. Add an expander showing the raw text.
4. Understand the pipeline: PDF → text extraction → sentence splitting → embedding → segmentation → segments. You'll build this step by step.
5. Practice Streamlit's caching: `@st.cache_resource` for loading the embedder (slow to load, should only happen once), `@st.cache_data` for caching segmentation results.

**Deliverable:** Working PDF upload and text extraction. Text displayed in the browser.

### Prep Week 2: VIM/CLI and Segmentation

**Goal:** Add the segmentation step and segment browser.

**Tasks:**
1. **VIM practice:** Edit the dashboard code to add new sections. Navigate, edit, save, auto-reload.
2. Wire up the OfflinePipeline: after text extraction, call `pipeline.process()` to segment the document.
3. Build a segment browser: expandable sections showing each segment's text, sentence count, and internal cost. Add segment statistics: total count, average size, cost distribution.
4. Time the segmentation: a 10-page PDF should segment in under 60 seconds on CPU. Display a progress indicator.
5. Manually check 5 random segments for coherence: does each segment cover a single topic?

**Deliverable:** Working segmentation with segment browser. Timing and quality checks done.

### Prep Week 3: Query and Retrieval

**Goal:** Add the query interface with pricing-guided retrieval.

**Tasks:**
1. Add a text input for queries. Wire up the PricingEngine to retrieve segments.
2. Display retrieved segments with their reduced costs. Highlight segments with negative reduced costs (included) vs positive (excluded).
3. Test with 5 different queries on your uploaded PDF. Verify retrieved segments are relevant.
4. Add pipeline step visualization: show which stage the system is at (PDF → Sentences → Segments → Query → Retrieve → Answer).
5. Optionally: add answer generation using the retrieved context as input to a small LLM, or just display the retrieved context as the "answer."

**Deliverable:** Full pipeline working: upload → segment → query → retrieve → display results.

---

## Build Phase (Weeks 4–10)

### Build Week 1 (Week 4): End-to-End Polish

**Tasks:**
- Clean up the pipeline visualization. Make each step clickable/expandable.
- Add a "regenerate" button that re-runs segmentation with different lambda values.
- Test on 2–3 different PDFs to verify robustness.
- Add answer display: either use an LLM to generate answers from context, or simply concatenate the retrieved segments as the "answer."

**Expected output:** Polished end-to-end demo working on multiple PDFs.

### Build Week 2 (Week 5): Top-K Cosine and Fixed-Chunk Baselines

**Tasks:**
- Implement top-k cosine retrieval: embed all segments, embed the query, return the k most similar segments by cosine similarity.
- Implement fixed-chunk retrieval: split the raw text into 512-word chunks (ignoring segment boundaries), embed each chunk, return the top-k by cosine similarity.
- Add both as retrieval options in the UI alongside pricing-guided retrieval.

**Expected output:** Both baselines working and selectable in the UI. Different methods return different segments for the same query.

### Build Week 3 (Week 6): BM25 Baseline and Side-by-Side Comparison

**Tasks:**
- Implement BM25 retrieval using the rank_bm25 library. Tokenize segments by whitespace, build the BM25 index, score queries.
- Build a side-by-side comparison UI: for the same query, show all 4 methods' results in parallel columns.
- Test on 5 queries. Observe: BM25 matches exact keywords while cosine matches semantics. Pricing-guided retrieval returns fewer, more targeted segments.

**Expected output:** All 4 retrieval methods working. Side-by-side comparison clearly showing differences.

### Build Week 4 (Week 7): Quantitative Evaluation Setup

**Tasks:**
- Create or find 50+ question-answer pairs for your test PDF. These can be factual questions where you know the answer is in the document.
- Define evaluation metrics: (1) number of segments retrieved, (2) retrieval precision (fraction of retrieved segments that are actually relevant — assess manually for 20 queries), (3) answer quality if using an LLM.
- Run all 4 retrieval methods on the 50 questions. Record results.

**Expected output:** 50+ QA pairs. Results for all 4 methods on all questions.

### Build Week 5 (Week 8): Lambda Sensitivity and Analysis

**Tasks:**
- Sweep lambda from 0.001 to 0.5 for the pricing-guided retrieval. At each lambda, measure precision and segment count.
- Plot the lambda sensitivity curve: lambda vs precision and lambda vs segment count.
- Identify the "sweet spot" lambda that balances precision and efficiency.
- Compare: at the sweet spot lambda, how does pricing-guided compare to the baselines?

**Expected output:** Lambda sensitivity curve. Sweet spot identified. Comparison at optimal lambda.

### Build Week 6 (Week 9): Segmentation vs Chunking Analysis

**Tasks:**
- Compare DP segmentation (topic-aware boundaries) vs fixed chunking (arbitrary boundaries) as a pre-processing step.
- Run top-k cosine retrieval on both segment types. Does topic-aware segmentation improve retrieval precision?
- Analyze specific examples where segmentation helps (a relevant passage is split across two fixed chunks but contained in one DP segment) and where it doesn't matter.

**Expected output:** Comparison of segmentation strategies with specific examples showing the difference.

### Build Week 7 (Week 10): Written Comparison Report

**Tasks:**
- Write a 2–3 page report covering: the demo pipeline, the 4 retrieval methods, quantitative comparison, lambda sensitivity, and segmentation vs chunking.
- Include: results table, lambda sensitivity curve, specific examples where pricing-guided retrieval outperforms baselines and where baselines are competitive.
- Make 3+ concrete findings: e.g., "Pricing-guided retrieval retrieves 40% fewer segments than top-k cosine while maintaining the same precision" or "BM25 outperforms embedding-based methods for queries with rare technical terms."
- Recommend a default lambda value backed by the sensitivity curve.

**Expected output:** Written report with data-backed findings and lambda recommendation.

---

## What "Done" Looks Like

1. A working demo app: PDF upload → segmentation → query → retrieval → answer display
2. Segment browser with statistics and pipeline visualization
3. Three baseline retrievers implemented and working (top-k cosine, fixed-chunk, BM25)
4. Side-by-side comparison UI showing all 4 methods
5. Quantitative evaluation on 50+ queries with precision metrics
6. Lambda sensitivity analysis with recommended default
7. Written report with 3+ concrete, data-backed findings about retrieval strategy performance

# S3: TypeScript Orchestrator Integration — Full Project Guide

## Project Classification: Software Engineering/Coding-Heavy

**Tier:** Advanced | **GPU:** None | **Team Size:** 2–3 | **Total Duration:** 10 weeks (3 prep + 7 build)

**Dependencies:** None (works against a mock server)

---

## Project Summary

You are building the TypeScript client that drives the multi-agent orchestrator. The Python ToolServer already exists (FastAPI with endpoints for reasoning, retrieval, code execution, and task decomposition). You build the client that receives a user goal, calls the server to decompose it into a task DAG, schedules tasks with topological sort to identify parallelism, dispatches them via HTTP with retries and timeouts, synchronizes results through shared memory, and handles partial failures gracefully.

---

## Motivation: Why This Matters for SKA-Agent

SKA-Agent's architecture is fundamentally multi-agent: a retrieval specialist (Jamba+SKA), a code executor, a reasoning coordinator (Qwen), and a parser collaborate on complex queries. The router decides which specialists to invoke, but the existing Python implementation dispatches them sequentially — one action at a time. This is a bottleneck. When the router determines that a MULTI_DOC query needs both "Retrieve FY2022 data" and "Retrieve FY2023 data" before "Compare years," the two retrieval tasks have no dependency on each other and could run in parallel. The sequential Python router wastes wall-clock time waiting for one to finish before starting the other.

The TypeScript orchestrator you're building sits above the Python ToolServer and unlocks parallelism. It receives a goal, calls the server's /tools/decompose endpoint to get a task DAG, uses topological sort to identify which tasks can run concurrently, and dispatches them via Promise.all. This is more than a performance optimization — it changes the system's practical capabilities. A 4-step sequential pipeline that takes 20 seconds becomes a 2-level parallel pipeline that takes 10 seconds, making multi-hop queries feasible in interactive settings.

The shared memory synchronization is particularly important here. In SKA-Agent, specialists communicate through the spectral memory protocol — the retriever writes operator updates that the reasoner can query. Your TS orchestrator must ensure that when a task at level 2 depends on results from level 1, those results have been written to shared memory before the dependent task reads them. This is the client-side coordination that makes the spectral memory protocol work across parallel agent execution, rather than just sequential handoffs.

---

## Starting Requirements

### Technical Prerequisites (Coding-Focused)

This is a software engineering project. The prep phase focuses on tools and patterns, not math.

- **TypeScript fundamentals:** Interfaces, generics, async/await, Promises, error handling with try/catch, type narrowing. If you're coming from Python, invest serious time here — TypeScript's type system is central to writing correct concurrent code.
- **Node.js and npm:** Project setup, package.json, tsconfig.json, running scripts, importing modules. You should be comfortable starting a project from scratch.
- **HTTP clients:** Using `fetch()` in Node.js (available natively in Node 18+), handling JSON request/response bodies, status codes, and error responses.
- **Concurrency with Promise.all:** Dispatching multiple HTTP requests in parallel and collecting results. Understanding that Promise.all fails fast (if one rejects, all reject) and how to handle this with Promise.allSettled or per-task error catching.
- **Graph algorithms:** Topological sort (Kahn's algorithm). You need to implement it, not just use a library. Understand in-degree counting, BFS processing, and cycle detection.
- **Testing:** Writing unit tests in TypeScript (using Jest or similar). Mocking HTTP responses for deterministic testing.

### Tools to Install During Prep

- Node.js 18+ (for native fetch support)
- TypeScript 5+ and ts-node (for running .ts files directly)
- A testing framework (Jest with ts-jest, or Vitest)
- Optional: nodemon for auto-reloading during development

### Codebase Familiarity

- `ska_agent/orchestration/__init__.py` — the Python ToolServer you'll communicate with. Read the endpoint definitions to understand the API contract: what each endpoint expects as input and returns as output.
- `ska_agent/core/structures.py` — the data structures that the server returns (modes, actions, cost vectors). Your TypeScript types should mirror these.

---

## Prep Phase (Weeks 1–3)

### Prep Week 1: TypeScript and Node.js Setup

**Goal:** Get a working TypeScript development environment and learn the language features you'll need.

**Tasks:**
1. Set up a new TypeScript project: `npm init`, install `typescript` and `ts-node`, create `tsconfig.json` with strict mode enabled.
2. Practice TypeScript fundamentals: define interfaces for TaskSpec, TaskResult, ScheduleLevel (from the starter code). Write functions that take and return these types.
3. Practice async/await: write a function that fetches a URL, parses JSON, and handles errors. Use `Promise.all` to fetch 3 URLs in parallel. Use `Promise.allSettled` to handle partial failures.
4. Set up a testing framework: write a test file, run it, verify assertions work.
5. Learn about AbortController for implementing request timeouts.

**Deliverable:** A working TypeScript project that compiles, runs, and has at least one passing test. A script that demonstrates parallel HTTP fetching.

### Prep Week 2: VIM, Command Line, and Graph Algorithms

**Goal:** Get comfortable with the development workflow and implement the core scheduling algorithm.

**Tasks:**
1. **VIM basics** (if not already familiar): Open files, navigate (hjkl, w/b, gg/G), edit (i/a/o, dd, yy, p), save/quit (:w, :q, :wq), search (/pattern). Practice editing TypeScript files in VIM. You don't need to be an expert — you need to be able to make edits without frustration.
2. **Command line workflow:** Running `tsc` to compile, `ts-node` to execute, `npm test` to test. Setting up a `Makefile` or npm scripts for common tasks.
3. **Implement Kahn's algorithm** for topological sort. The algorithm: (a) compute in-degree for each node, (b) add all zero-in-degree nodes to a queue, (c) process the queue: remove a node, decrease in-degrees of its dependents, add any new zero-in-degree nodes to the queue. Group nodes by "level" (all nodes processed in the same BFS wave can run in parallel).
4. **Test the scheduler** on 5 different DAG topologies: linear chain (A→B→C), fork (A→B, A→C), join (A→C, B→C), diamond (A→B, A→C, B→D, C→D), and cyclic (A→B→A, should throw an error).

**Deliverable:** A working `buildSchedule()` function that correctly identifies parallel groups and detects cycles. 5 passing unit tests.

### Prep Week 3: Mock Server and HTTP Dispatch

**Goal:** Build the mock server and agent pool so you can test without the real Python server.

**Tasks:**
1. Implement the mock server (starter code provided). It should respond to all 6 endpoints (/tools/decompose, /tools/retrieve, /tools/reason, /tools/execute, /memory/write, /memory/read, /health) with realistic mock responses.
2. Implement the AgentPool with dispatch(), retry logic, exponential backoff, and timeout handling.
3. Test dispatch against the mock server: successful requests, retries on 500 errors, timeout after configured duration.
4. Write an integration test: start mock server, dispatch a task, verify the response.

**Deliverable:** A mock server that responds to all endpoints. An AgentPool that handles retries and timeouts. Integration test passing.

---

## Build Phase (Weeks 4–10)

### Build Week 1 (Week 4): Basic Orchestrator

**Tasks:**
- Implement the Orchestrator class that chains: decompose (call /tools/decompose) → schedule (Kahn's algorithm) → dispatch (parallel tasks per level) → synthesize (call /tools/reason with all results).
- End-to-end test against mock server: "Compare FY2022 and FY2023 spending" → decompose into tasks → schedule → dispatch → synthesize → return answer.
- Verify parallelism: time the execution. If two tasks are at the same level, total wall-clock should be less than the sum of individual latencies (proving they ran in parallel).

**Expected output:** Orchestrator running end-to-end against mock server. Parallelism demonstrated by timing.

### Build Week 2 (Week 5): Shared Memory Synchronization

**Tasks:**
- Implement `shared_memory.ts`: a client-side text store that syncs with the Python /memory/write and /memory/read endpoints.
- After each task completes, write its result to shared memory so subsequent tasks can read it.
- Test: write a value, read it back, verify match. Verify that a task at level 2 can read the output of a level 1 task via shared memory.

**Expected output:** Shared memory round-trips working. Tasks can share results through the memory endpoint.

### Build Week 3 (Week 6): Error Handling and Partial Failure

**Tasks:**
- When a task fails permanently (all retries exhausted), mark it as failed and mark all downstream dependents as "skipped" (they can't run without their dependency).
- The orchestrator should synthesize a partial result from the tasks that did succeed, not crash.
- Test: inject a failure (mock server returns 500 permanently for one endpoint). Verify dependents are skipped, successful tasks complete, and synthesis uses only successful results.

**Expected output:** Graceful partial failure handling. No crashes, no hangs. Partial results synthesized.

### Build Week 4 (Week 7): Integration with Real Python Server

**Tasks:**
- Start the real Python ToolServer with `uvicorn ska_agent.orchestration:app --port 8741`.
- Run the orchestrator against it. The real server's /tools/decompose will return different (potentially more complex) task DAGs.
- Debug any differences between mock and real server responses (field names, data types, error formats).
- Get at least one end-to-end flow working against the real server.

**Expected output:** End-to-end execution against the real Python server. At least one goal fully processed.

### Build Week 5 (Week 8): Latency Tracking and Logging

**Tasks:**
- Add structured logging: each task records start time, end time, which schedule level it ran in, success/failure, and retry count.
- Compute per-level parallelism: for each level, how many tasks ran in parallel, and what was the level's wall-clock time?
- Compute overall speedup: total wall-clock time vs hypothetical sequential time (sum of all task latencies).
- Print a human-readable summary after each orchestration run.

**Expected output:** Summary log for each run showing per-level timing, parallelism achieved, and speedup. For a 4-task DAG with 2 parallel tasks, measured speedup is at least 1.5×.

### Build Week 6 (Week 9): Robustness and Polish

**Tasks:**
- Handle additional edge cases: empty task list (decompose returns nothing), single-task DAG, very large DAGs (20+ tasks), tasks that return very large payloads.
- Add configurable timeouts (per-task and global), configurable retry counts, and max parallelism limits.
- Write comprehensive tests for all error paths.

**Expected output:** Robust orchestrator that handles all edge cases. Configurable parameters. Full test coverage.

### Build Week 7 (Week 10): Documentation and Final Testing

**Tasks:**
- Write a README explaining how to set up, configure, and run the orchestrator.
- Write API documentation for all public interfaces (Orchestrator, AgentPool, SharedMemory, Scheduler).
- Run final integration tests against both mock and real servers.
- Clean up code: consistent naming, proper TypeScript types (no `any`), error messages, comments.

**Expected output:** Clean, documented codebase. README with setup instructions. All tests passing.

---

## What "Done" Looks Like

1. A TypeScript orchestrator that decomposes goals into task DAGs, schedules them with topological sort, dispatches parallel tasks via HTTP, and synthesizes results
2. Retry logic with exponential backoff and configurable timeouts
3. Shared memory synchronization between tasks
4. Graceful partial failure handling (skip dependents, synthesize from successes)
5. Integration tested against both mock and real Python ToolServer
6. Latency logging demonstrating parallelism and speedup
7. Clean, documented TypeScript codebase

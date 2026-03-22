CS195 Project 8: SKA vs Efficient Attention Methods

Weeks 1-2: Learning Phase (shared across all projects)
    Week 1: Read learning/week1_koopman_theory.md. Complete exercises 1-5.
    Week 2: Work through learning/week2_hands_on.md and week2_exercises.py.

Weeks 3-10: Project Phase

Overview:
    Implement linear attention, Performer, and Hedgehog alongside SKA.
    Benchmark all on the same retrieval tasks. Position SKA in the
    broader efficient attention landscape.

Hardware: single GPU. All methods run on small models.

Week 3: Implement LinearAttention with causal cumsum. Test on short sequences.
Week 4: Implement PerformerAttention with FAVOR+ random features.
Week 5: Implement CosineReweightedAttention (Hedgehog-style learned features).
Week 6: Build model builder that swaps in any method. Train all on MQAR.
Week 7: Evaluate all methods on MQAR, phonebook, induction at multiple lengths.
Week 8: Measure theoretical and empirical FLOPs and memory for each.
Week 9: Build Pareto plots (accuracy vs FLOPs, accuracy vs memory).
Week 10: Write final report positioning SKA against all baselines.

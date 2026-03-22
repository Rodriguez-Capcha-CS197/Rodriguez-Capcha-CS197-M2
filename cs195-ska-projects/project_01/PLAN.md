CS195 Project 1: Online SKA with Rank-1 Updates

Weeks 1-2: Learning Phase (shared across all projects)
    Week 1: Read learning/week1_koopman_theory.md. Complete exercises 1-5.
    Week 2: Work through learning/week2_hands_on.md and week2_exercises.py.

Weeks 3-10: Project Phase

Overview:
    Implement an incremental version of SKA that updates its sufficient
    statistics one token at a time using rank-1 Cholesky updates. Prove
    it matches the batch version's accuracy while achieving O(1) cost
    per token.

Hardware: CPU only. No GPU needed.

Week 3: Implement _rank1_cholesky_update. Unit test against recomputed cholesky.
Week 4: Implement process_token and query. Test on random data.
Week 5: Complete compare_batch_vs_online. Verify max difference < 1e-5.
Week 6: Complete benchmark_scaling. Show batch scales linearly, online stays constant.
Week 7: Complete eval_on_mqar. Show identical accuracy between batch and online.
Week 8: Numerical stability analysis. Test ill-conditioned Gram matrices.
Week 9: Implement rank-1 Cholesky downdate for sliding-window SKA (stretch goal).
Week 10: Write final report (4-6 pages) with scaling plot and accuracy tables.

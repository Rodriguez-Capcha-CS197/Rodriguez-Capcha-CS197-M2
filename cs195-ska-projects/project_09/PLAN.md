CS195 Project 9: Recurrent Koopman MLP

Weeks 1-2: Learning Phase (shared across all projects)
    Week 1: Read learning/week1_koopman_theory.md. Complete exercises 1-5.
    Week 2: Work through learning/week2_hands_on.md and week2_exercises.py.

Weeks 3-10: Project Phase

Overview:
    Extend the Koopman MLP from a static feedforward layer to a recurrent
    one by letting the eigenvalues evolve over the sequence. Creates a
    nonlinear recurrence with O(d_k) state vs TTT-MLP's O(d_k^2).

Hardware: single GPU for training small models.

Week 3: Understand static Koopman MLP. Run on MQAR, confirm it works.
Week 4: Design eigenvalue update network. Implement RecurrentKoopmanMLP (sequential).
Week 5: Test on MQAR. Does recurrence help vs static Koopman MLP?
Week 6: Implement SimpleTTTMLP baseline. Train and evaluate.
Week 7: Implement parallel approximation for faster training.
Week 8: Compare all four variants (SwiGLU, static Koopman, recurrent Koopman, TTT-MLP).
Week 9: Test across multiple context lengths and tasks.
Week 10: Write final report. Key finding: O(d_k) recurrent state vs O(d_k^2).

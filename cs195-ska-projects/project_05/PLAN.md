CS195 Project 5: Spectral Analysis of the Koopman Operator

Weeks 1-2: Learning Phase (shared across all projects)
    Week 1: Read learning/week1_koopman_theory.md. Complete exercises 1-5.
    Week 2: Work through learning/week2_hands_on.md and week2_exercises.py.

Weeks 3-10: Project Phase

Overview:
    Train SKA on different tasks, extract the learned operators,
    analyze eigenvalue spectra for interpretable structure.

Hardware: single GPU for training small models. Analysis on CPU.

Week 3: Modify SKA to expose intermediate A_w and B_v. Verify extraction works.
Week 4: Train on MQAR. Extract operator, compute eigenvalues, make first plot.
Week 5: Train on phonebook, induction, selective copy. Extract all operators.
Week 6: Compare spectra across tasks. Overlay eigenvalue plots.
Week 7: Head specialization analysis. Ablate individual heads.
Week 8: Track how spectrum changes during training (extract at checkpoints).
Week 9: Test whether eigenvalue structure predicts task performance.
Week 10: Write final report with eigenvalue visualizations.

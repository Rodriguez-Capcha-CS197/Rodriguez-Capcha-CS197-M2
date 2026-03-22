CS195 Project 3: Koopman MLP Ablation

Weeks 1-2: Learning Phase (shared across all projects)
    Week 1: Read learning/week1_koopman_theory.md. Complete exercises 1-5.
    Week 2: Work through learning/week2_hands_on.md and week2_exercises.py.

Weeks 3-10: Project Phase

Overview:
    Measure the effect of replacing SwiGLU MLPs with Spectral Koopman
    MLPs in a pretrained GPT-2 model. Quantify parameter savings vs
    perplexity cost, and how fast fine-tuning recovers quality.

Hardware: Colab free tier (T4 GPU, 16GB).

Week 3: Load GPT-2 small. Implement measure_perplexity. Record baseline.
Week 4: Design weight mapping from SwiGLU to Koopman MLP. Implement replace_mlp_layers.
Week 5: Measure perplexity of replaced model (both pure and gated, no training).
Week 6: Implement finetune. Train only Koopman params for 500 steps. Measure recovery.
Week 7: Per-layer sensitivity analysis: replace one layer at a time.
Week 8: Sweep expansion ratio. Plot parameter count vs perplexity.
Week 9: Compare pure vs gated variant across all experiments.
Week 10: Write final report with perplexity tables and Pareto plot.

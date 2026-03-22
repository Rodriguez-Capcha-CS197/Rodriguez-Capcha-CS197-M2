CS195 Project 7: Hybrid Context Switching

Weeks 1-2: Learning Phase (shared across all projects)
    Week 1: Read learning/week1_koopman_theory.md. Complete exercises 1-5.
    Week 2: Work through learning/week2_hands_on.md and week2_exercises.py.

Weeks 3-10: Project Phase

Overview:
    Build a layer that dynamically switches between attention and SKA
    based on context length. Find the crossover point on real hardware.

Hardware: Colab with GPU (T4 minimum, A100 preferred for longer contexts).

Week 3: Implement HybridAttentionLayer with hard threshold switching.
Week 4: Implement CUDA timing benchmark for attention at various seq_lens.
Week 5: Implement CUDA timing benchmark for SKA. Find raw crossover point.
Week 6: Implement memory benchmark. Find memory crossover point.
Week 7: Implement smooth blending variant. Train on MQAR.
Week 8: Test accuracy preservation across the switch point.
Week 9: Try different threshold strategies (fixed, learned, schedule).
Week 10: Write final report with latency and memory plots.

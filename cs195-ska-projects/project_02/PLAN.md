CS195 Project 2: SKA as Drop-In for Pretrained Attention

Weeks 1-2: Learning Phase (shared across all projects)
    Week 1: Read learning/week1_koopman_theory.md. Complete exercises 1-5.
    Week 2: Work through learning/week2_hands_on.md and week2_exercises.py.

Weeks 3-10: Project Phase

Overview:
    Load a small pretrained LLM, replace its attention layers with SKA
    initialized from the original Q/K/V weights, show retrieval accuracy
    is preserved with lower memory. No training required.

Hardware: Colab free tier (T4) for 0.5B model, Colab Pro (A100) for longer context.

Week 3: Load Qwen2.5-0.5B-Instruct. Map architecture. Implement extract_attention_weights.
Week 4: Design GQA-to-SKA weight mapping. Implement init_ska_from_attention.
Week 5: Implement replace_attention_layer. Verify model still generates text.
Week 6: Implement needle_in_haystack_test. Run on original model for baseline.
Week 7: Run needle_in_haystack on SKA model. Compare accuracy + memory.
Week 8: Push to longer context (8K, 16K, 32K). Plot memory scaling.
Week 9: Try multiple SKA ranks. Find minimum rank preserving accuracy.
Week 10: Write final report with accuracy table and memory scaling plot.

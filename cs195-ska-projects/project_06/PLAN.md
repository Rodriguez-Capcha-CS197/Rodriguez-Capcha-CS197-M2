CS195 Project 6: SKA for Vision Transformers

Weeks 1-2: Learning Phase (shared across all projects)
    Week 1: Read learning/week1_koopman_theory.md. Complete exercises 1-5.
    Week 2: Work through learning/week2_hands_on.md and week2_exercises.py.

Weeks 3-10: Project Phase

Overview:
    Test whether SKA generalizes beyond language by replacing attention
    in a pretrained ViT with SKA and evaluating on image classification.
    ViT attention is bidirectional (not causal), so prefix_mask is all-ones.

Hardware: Colab T4 (16GB) for ViT-Tiny/Small.

Week 3: Load ViT-Tiny. Understand architecture. Run baseline eval on CIFAR-100.
Week 4: Extract attention weights. Study bidirectional vs causal difference.
Week 5: Implement build_ska_vit_layer. Handle non-causal setting.
Week 6: Replace all attention layers. Measure accuracy without fine-tuning.
Week 7: Fine-tune SKA parameters only for 1-2 epochs. Measure recovery.
Week 8: Sweep SKA rank. Find minimum rank for <1% accuracy drop.
Week 9: Visualize what the Koopman operator captures in patch space.
Week 10: Write final report. Key question: does Koopman learn spatial relationships?

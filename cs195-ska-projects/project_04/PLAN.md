CS195 Project 4: SKA Rank vs Retrieval Accuracy

Weeks 1-2: Learning Phase (shared across all projects)
    Week 1: Read learning/week1_koopman_theory.md. Complete exercises 1-5.
    Week 2: Work through learning/week2_hands_on.md and week2_exercises.py.

Weeks 3-10: Project Phase

Overview:
    Characterize the tradeoff between SKA rank and retrieval accuracy
    on synthetic tasks. Find the Pareto frontier.

Hardware: single GPU or CPU. Small models, short context.

Week 3: Build parameterized model builder. Train at rank=32 on MQAR.
Week 4: Implement memory and FLOPs measurement. Verify formulas by hand.
Week 5: Run full rank sweep [8, 16, 24, 32, 48, 64, 96, 128] on MQAR.
Week 6: Run same sweep on phonebook and induction tasks.
Week 7: Run sweep across context lengths [512, 1024, 2048, 4096].
Week 8: Plot Pareto frontiers. Identify minimum rank for >95% accuracy per task.
Week 9: Add attention baseline at each point for comparison.
Week 10: Write final report with Pareto plots and recommendation table.

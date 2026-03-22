CS195 Project 10: SKA for Retrieval-Augmented Generation

Weeks 1-2: Learning Phase (shared across all projects)
    Week 1: Read learning/week1_koopman_theory.md. Complete exercises 1-5.
    Week 2: Work through learning/week2_hands_on.md and week2_exercises.py.

Weeks 3-10: Project Phase

Overview:
    Frame SKA as a learned retrieval mechanism. Encode a document corpus
    into a Koopman operator (fixed-size state), query it with hidden
    states. Compare retrieval quality and memory vs FAISS.

Hardware: CPU for FAISS, single GPU for encoding.

Week 3: Set up sentence transformer encoding. Encode a small corpus.
Week 4: Implement FAISSRetriever baseline. Verify on a QA dataset.
Week 5: Implement SKARetriever.add_documents with incremental Gram updates.
Week 6: Implement SKARetriever.retrieve. Test on same QA dataset.
Week 7: Compare recall@5 and MRR at various corpus sizes.
Week 8: Sweep SKA rank. Plot retrieval quality vs memory.
Week 9: Scale to larger corpus (5K-10K docs). Show SKA memory stays constant.
Week 10: Write final report with retrieval quality and memory comparison.

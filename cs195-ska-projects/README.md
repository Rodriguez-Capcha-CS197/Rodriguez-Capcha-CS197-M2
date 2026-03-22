CS195: Spectral Koopman Attention - Student Projects

This repository contains learning materials and 10 independent research
projects exploring Spectral Koopman Attention (SKA) and related
Koopman-theoretic methods for efficient sequence modeling.

Setup:
    pip install torch transformers datasets matplotlib

    Some projects need additional packages:
    Project 6: pip install timm
    Project 10: pip install faiss-cpu sentence-transformers

    All projects import from the shared/ directory. Run from the
    project directory:
        cd project_01
        python online_ska.py

Structure:
    learning/            Weeks 1-2 learning materials (everyone reads these)
        week1_koopman_theory.md    Koopman theory from first principles
        week2_hands_on.md          Hands-on coding guide
        week2_exercises.py         Runnable exercises with TODOs

    shared/              Shared library (SKA, Koopman MLP, eval tasks, utilities)

    project_01/          Online SKA with rank-1 Cholesky updates
    project_02/          Weight transplant from pretrained attention to SKA
    project_03/          Koopman MLP ablation on GPT-2
    project_04/          SKA rank vs retrieval accuracy tradeoffs
    project_05/          Spectral analysis of the learned Koopman operator
    project_06/          SKA for Vision Transformers
    project_07/          Hybrid context switching (attention <-> SKA)
    project_08/          SKA vs linear attention vs Performer vs Hedgehog
    project_09/          Recurrent Koopman MLP
    project_10/          SKA for retrieval-augmented generation

Quarter timeline:
    Weeks 1-2:  Learning phase (shared across all projects)
                Everyone reads the learning materials and completes the
                exercises. By end of week 2 you should be able to explain
                every line in ska.py and koopman_mlp.py.

    Weeks 3-10: Project phase (each student works on their chosen project)
                Follow the PLAN.md in your project directory.
                One deliverable per week.
                Final report due week 10.

Each project contains:
    PLAN.md              Weekly plan with milestones and deliverables
    *.py                 Starter code with TODO markers where you fill in

GPU requirements:
    No GPU:     Projects 1, 4 (small scale), 5 (analysis only)
    Colab T4:   Projects 2, 3, 6, 7, 8, 9
    Colab A100: Project 2 (longer context), Project 7 (longer context), 10

How to work on a project:
    1. Weeks 1-2: read learning materials, do exercises, ask questions
    2. Week 3: read PLAN.md for your project, read starter code
    3. Search for TODO in the starter code -- these are what you implement
    4. Follow the weekly plan, one deliverable per week
    5. The shared/ library provides all the core modules you need

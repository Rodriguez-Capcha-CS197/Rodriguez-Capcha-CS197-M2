Week 2: Hands-On with SKA and Koopman MLP

This week you run code, break things, and build intuition for how
the modules work in practice. By the end of this week you should be
able to explain every line in ska.py and koopman_mlp.py.

All exercises run on CPU. No GPU needed.


Part A: SKA from Scratch

Exercise 1: Build a minimal SKA by hand.

    Open a Python REPL or notebook. Do not import SKAModule.
    Using only torch and torch.linalg, implement the following
    for a single head (H=1) with rank r=4 and d_model=8:

    a. Create random W_key (8, 4), W_query (8, 4), W_value (8, 6)
    b. Create a random input sequence x of shape (1, 20, 8)
    c. Project: z = x @ W_key.T, v = x @ W_value.T
    d. Compute G = z^T z (sum of outer products over time)
    e. Compute M = z[1:]^T z[:-1]
    f. Compute C_v = v^T z
    g. Add ridge: G_tilde = G + 0.001 * I
    h. Cholesky: L = torch.linalg.cholesky(G_tilde)
    i. Solve for A_w: A_w = M @ (G_tilde)^{-1}
       Hint: use torch.cholesky_solve
    j. Solve for B_v: B_v = C_v @ (G_tilde)^{-1}
    k. For a query token z_q, compute: output = B_v @ A_w @ z_q

    Print the output shape. It should be (6,) -- the value dimension.

Exercise 2: Compare your manual implementation to SKAModule.

    Now import SKAModule from shared/ska.py. Create one with the same
    dimensions (d_model=8, n_heads=1, rank=4, head_dim=6). Copy your
    W_key, W_query, W_value into the module's weights. Run forward on
    the same input. Compare outputs. They should be close (differences
    from spectral normalization and power filtering).

Exercise 3: Visualize the Gram matrix.

    Generate three different types of sequences:
    a. Random noise (torch.randn)
    b. Repeating pattern ([1,2,3,4] repeated 5 times, embedded)
    c. All-same token (same vector repeated 20 times)

    For each, compute G and visualize it as a heatmap (use matplotlib
    or just print the values). How does G look different for each?
    What does this tell you about how well-conditioned the ridge
    regression will be?

Exercise 4: Eigenvalue exploration.

    Take the A_w operator from Exercise 1. Compute its eigenvalues
    with torch.linalg.eig. Plot them in the complex plane (real part
    on x-axis, imaginary part on y-axis). Draw the unit circle.

    Now apply spectral normalization (divide A_w by its largest
    singular value). Recompute eigenvalues and plot again. All
    eigenvalues should be inside the unit circle now.

    Try A_w^2 and A_w^4. How do the eigenvalues change? Why does
    power filtering amplify some modes and suppress others?


Part B: Koopman MLP

Exercise 5: Understand the 2x2 rotation blocks.

    The Koopman MLP uses block-diagonal 2x2 rotation matrices:

        [gamma, omega]   [g1]     [gamma*g1 + omega*g2]
        [-omega, gamma] * [g2]  =  [-omega*g1 + gamma*g2]

    a. Set gamma=1, omega=0. What does this rotation do? (Identity)
    b. Set gamma=0, omega=1. What does this rotation do? (90-degree rotation)
    c. Set gamma=0.7, omega=0.7. What angle is this? What is the
       magnitude (sqrt(gamma^2 + omega^2))?
    d. When spectral normalization clamps the radius to <= 1, what
       does this mean geometrically for the rotation?

    Implement this in Python: create a (2,) vector, apply the rotation
    with several gamma/omega values, and plot the trajectory.

Exercise 6: Trace through SpectralKoopmanMLP.

    Open shared/koopman_mlp.py. For d=8 and expand=2.667:
    a. What is d_k? (Compute by hand, then check)
    b. How many parameters in lift? In readout? In gamma+omega?
    c. Create the module: mlp = SpectralKoopmanMLP(8)
    d. Print total parameter count
    e. Create a SwiGLU MLP with the same d and expand
    f. Print its parameter count
    g. What is the ratio? Does it match the ~1/3 savings claim?

Exercise 7: Koopman MLP vs SwiGLU on random data.

    Create both modules with d=64. Initialize both randomly.
    Feed the same random input (1, 100, 64) through each.
    Compute the output variance of each. The Koopman MLP with
    gamma=1, omega~N(0,0.1) should start close to identity
    (output close to input). SwiGLU should start with larger
    output variance. Why?


Part C: Putting It Together

Exercise 8: Build a tiny model.

    Using SmallTransformerLM from shared/utils.py, build a model
    with vocab_size=512, d_model=64, n_layers=4, n_heads=4.

    a. First with all attention layers (ska_layer_indices=[])
    b. Then with SKA in layers 1 and 3 (ska_layer_indices=[1,3])
    c. Then with SKA + Koopman MLP (use_koopman_mlp=True)

    Print parameter counts for all three. Generate random token
    sequences and verify all three produce logits of the right shape.

Exercise 9: Train on MQAR.

    Using the MQARDataset from shared/eval_tasks.py, create a small
    dataset: n_examples=500, M=4, seq_len=128, vocab_size=512.

    Train your three models from Exercise 8 for 1000 steps each.
    Use the evaluate function from shared/utils.py to measure
    retrieval accuracy after training.

    Questions to answer:
    a. Do all three converge?
    b. Which converges fastest?
    c. What is the final accuracy of each?
    d. What happens if you increase M to 8 or 16?

Exercise 10: Prefix mask experiment.

    Using the same MQAR setup, train the SKA model twice:
    a. With the correct prefix_mask (1 for KV pairs + distractors, 0 for queries)
    b. With prefix_mask = all ones (every token is "prefix")

    Compare accuracy. The correct prefix mask should help because it
    tells SKA exactly which tokens to build the operator from. With
    all-ones, the query tokens contaminate the operator estimation.

    Then try with prefix_mask = None (the default in SKAModule, which
    falls back to all-ones). Does it match case (b)? It should.


Suggested Reading (Optional)

    Koopman, B.O. (1931). Hamiltonian systems and transformation in
    Hilbert space. PNAS.
    (The original 2-page paper. Short and readable.)

    Brunton, S. et al. (2016). Discovering governing equations from
    data by sparse identification of nonlinear dynamical systems. PNAS.
    (Modern Koopman methods for dynamical systems.)

    Sun, Y. et al. (2024). Learning to (Learn at Test Time): RNNs
    with Expressive Hidden States.
    (TTT paper. SKA solves the same problem they solve with gradient
    descent, but in closed form.)

    Arora, S. et al. (2024). Zoology: Measuring and Improving Recall
    in Efficient Language Models.
    (Defines MQAR and other retrieval benchmarks for efficient models.)

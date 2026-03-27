# Week 1: Koopman Operator Theory for Sequence Modeling
This document teaches you the math behind Spectral Koopman Attention
from scratch. No prior knowledge of dynamical systems is assumed.
You should be comfortable with linear algebra (matrix multiplication,
eigenvalues, least squares) and basic machine learning (neural networks,
attention mechanism).


## 1. The Problem: Why Attention is Expensive

In a transformer, self-attention computes:

$$
\text{Attention}(Q, K, V) = \text{softmax}(Q K^T / sqrt(d)) V
$$

For a sequence of T tokens, $Q K^T$ is a ($T \times T$) matrix.
Computing it costs $O(T^2)$ FLOPs, and storing the KV cache for generation costs $O(T)$ memory per layer.
At 128K tokens, this becomes the bottleneck.

**The question:** can we compute something that behaves like attention (retrieve relevant values given a query) without the $O(T)$ per-token cost?

## 2. What is a Koopman Operator?

Imagine you have a dynamical system.
A ball bouncing, a pendulum swinging, or tokens flowing through a neural network.
At each timestep, the system has a state $x_t$, and there's some (possibly nonlinear) function $f$ that takes you to the next state:

$$
x_{t+1} = f(x_t)
$$

The Koopman operator is a trick for turning nonlinear dynamics into linear dynamics.
**The idea:** instead of tracking $x_t$ directly, track "observable functions" of $x_t$.
An observable is just any function $g(x)$ that measures something about the state.

If you pick the right set of observables, the dynamics become linear in observable space:

$$
g(x_{t+1}) = K g(x_t)
$$

where $K$ is a matrix called the Koopman operator.
This is exact (not an approximation) if you use infinitely many observables.
In practice we use a finite set and get a good approximation.

Why does this help? Because linear dynamics are easy.
If you know $K$, you can predict the state at any future time by matrix multiplication:

$$
g(x_{t+n}) = K^n g(x_t)
$$

No need to simulate step by step.

## 3. From Dynamical Systems to Attention

Now think of the token sequence in a language model as a dynamical system.
The "state" at position $t$ is the residual stream $h_t$ (the hidden representation at that position).
The "dynamics" are whatever transformation moves information from one position to the next.

**In this framing:**
    - The "observables" are learned projections of the residual stream
      (analogous to keys in attention)
    - The Koopman operator $K$ captures how information flows across
      positions
    - "Querying" the operator is like doing attention: you project
      your query into observable space, apply $K$, and read out values

**The key insight:** once $K$ is estimated, applying it to a query is $O(r^2)$ where $r$ is the rank of the observable space.
This doesn't depend on $T$ at all.
You pay $O(T)$ once to estimate $K$ from the sequence, then every query is $O(r^2)$ regardless of how long the context was.

## 4. How SKA Estimates the Operator

Given a sequence of hidden states $h_1, \dots, h_T$, SKA does the following:

**Step 1:** Project into observable space.

$$
\begin{split}
z_t & = W_{\text{key}} h_t \quad (\text{key projection}, d_{model} \rightarrow r) \\
v_t & = W_{\text{val}} h_t \quad (\text{value projection}, d_{model} \rightarrow P)
\end{split}
$$

**Step 2:** Build the Gram matrix (sufficient statistics).

$$
G = \sum_t z_t z_t^T \quad (r \times r \text{matrix, measures "how much data we have"})
$$

**Step 3:** Build the transition covariance.

$$
M = \sum_t z_{t+1} z_t^T  (r \times r, \text{measures "how states evolve"})
$$

**Step 4:** Build the value readout.

$$
C_v = sum_t v_t z_t^T     (P \times r, \text{maps from key space to value space})
$$

**Step 5:** Solve for the operator via ridge regression.

$$
\begin{split}
A_w & = M (G + \lambda I)^{-1} \quad (\text{the Koopman transition operator})\\
B_v & = C_v (G + \lambda I)^{-1} \quad (\text{the value readout operator})
\end{split}
$$

The ridge regularization ($\lambda I$) prevents overfitting when $G$ is ill-conditioned (when some directions in key space have little data).

**Step 6:** Apply to queries.

$$
\begin{split}
z_q & = W_query h_query \quad (\text{project query into observable space})\\
output & = B_v A_w^K z_q \quad (\text{apply operator K times, read out values})
\end{split}
$$

The power $K$ in $A_w^K$ is a hyperparameter.
$K=1$ is a simple one-step prediction.
$K=2$ amplifies the dominant modes.
Think of it as asking "what would happen if we ran the dynamics forward $K$ steps?"

## 5. Why Ridge Regression (not Gradient Descent)?

If you squint, this looks a lot like linear regression.
We have input-output pairs ($z_t, z_{t+1}$) and we want to find the matrix $A$ that best predicts $z_{t+1}$ from $z_t$.
Ridge regression gives the exact closed-form solution:

$$
A = (\sum z_{t+1} z_t^T) (\sum z_t z_t^T + \lambda I)^{-1}
$$

This is the same as what you'd get if you ran gradient descent on the squared error loss $\|A z_t - z_{t+1}\|^2$ for infinitely many steps with the right learning rate.
But ridge regression gets there in one step.

This is exactly the connection to Test-Time Training (TTT).
TTT-Linear approximates this same solution via gradient descent.
SKA computes it directly.
Same destination, different paths.

## 6. Spectral Normalization: Keeping Things Stable

The operator $A_w$ might have eigenvalues with magnitude > 1, which would make $A_w^K$ explode.
SKA prevents this with spectral normalization:
estimate the largest singular value of $A_w$ using power iteration, then divide by it so all eigenvalues have magnitude <= 1.

This is the same idea as spectral normalization in GANs, applied to the Koopman operator instead of a discriminator weight matrix.

## 7. The Cholesky Factorization

Instead of directly computing $(G + \lambda I)^{-1}$, SKA uses the Cholesky decomposition: $G + \lambda I = L L^T$ where $L$ is lower triangular.
Then:

$$
(G + \lambda I)^{-1} x = L^{-T} L^{-1} x
$$

Solving triangular systems is $O(r^2)$ instead of $O(r^3)$ for general matrix inversion.
This matters when you have many queries.

## 8. Connection to Attention

**Standard attention:**

$$
\text{output}_i = \text{sum}_j \text{softmax}(q_i . k_j / \text{sqrt}(d)) v_j
$$

This looks at every key $k_j$ for every query $q_i$.
Cost: $O(T)$ per query.

SKA:

$$
\text{output}_i = B_v A_w^K L^{-1} W_{\text{query}} h_i
$$

This applies a fixed operator to the query.
Cost: $O(r^2)$ per query.

The operator $(B_v, A_w, L)$ encodes everything the model learned from the context.
It's a compressed representation of the key-value store.

## 9. State Size Comparison

Attention KV cache at 128K context, 32 heads, head_dim=128:
> 2 * 128K * 32 * 128 * 2 bytes = ~2GB per layer

SKA state with rank=64, 32 heads:
> 32 * (3 * 64^2 + 128 * 64) * 2 bytes = ~2.6MB per layer

_That's roughly 800x smaller._

## 10. Exercises

**Exercise 1:**
* z_1 = [1, 0] and z_2 = [0, 1].
* Compute $G$, $M$, and $A_w$ by hand (with `lambda=0`).
* What does the operator do?
* What happens when you apply A_w to z_1?

**Exercise 2:**
* `z_1 = [1, 0]`, `z_2 = [1, 1]`, `z_3 = [1, 2]`
* compute $G, M, A_w$.
* What dynamics does this capture? What does $A_w^2 z_1$ predict?

**Exercise 3:**
* Open `shared/ska.py`.
* Trace through the forward method line by line.
* For each line, write a comment explaining what it computes and what shape the tensor has.
* Check your understanding by adding assert statements for shapes and running on a dummy input.

**Exercise 4:**
* Modify `SKAModule`.
* forward to return the eigenvalues of $A_w$ alongside the output.
* Run it on a random input and print the eigenvalues.
* Are they all inside the unit circle? What happens if you remove the spectral normalization?

**Exercise 5:**
* Take the MQAR dataset `from shared/eval_tasks.py`.
* Generate one example with M=4 key-value pairs.
* Manually trace what the "ideal" Koopman operator should look like for this example:
* if the keys are the inputs and the values are the outputs, what matrix A_w would perfectly retrieve all values?

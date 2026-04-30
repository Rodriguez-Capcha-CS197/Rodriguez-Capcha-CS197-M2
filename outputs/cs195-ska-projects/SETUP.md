# Setup Guide

Follow these steps before starting the Week 1 learning materials. The whole process takes about 5 minutes.

## 1. Prerequisites

- **Python 3.10+** is required. Check with `python3 --version`.
- **Git** for cloning the repository.
- No GPU needed for setup or for the first two weeks of learning materials.

## 2. Clone and Create a Virtual Environment

```bash
git clone <your-repo-url> cs195-ska-projects
cd cs195-ska-projects

python3 -m venv venv
source venv/bin/activate        # macOS / Linux
# venv\Scripts\activate         # Windows
```

## 3. Install Dependencies

**Core packages** (needed by all projects and learning materials):

```bash
pip install --upgrade pip
pip install torch numpy scipy matplotlib
pip install transformers sentence-transformers
```

**Additional packages** (needed by some projects — safe to install now):

```bash
pip install streamlit plotly          # A1, A2, A3, A4 (dashboards)
pip install scikit-optimize rank_bm25 # S4 (Bayesian opt), A3 (BM25 baseline)
pip install datasets                  # training data utilities
```

**TypeScript** (only for S3):

```bash
# Only if you're working on the TypeScript Orchestrator project (S3)
# Requires Node.js 18+
npm install typescript ts-node
```

**Jupyter** (optional, for interactive exploration):

```bash
pip install jupyter ipykernel
```

## 4. Verify the Setup

Run this from the `cs195-ska-projects/` directory:

```bash
python3 -c "
import torch
import numpy as np
from shared.ska import SKAModule
from shared.koopman_mlp import KoopmanMLP
print(f'PyTorch {torch.__version__}')
print(f'NumPy {np.__version__}')
print('shared imports OK')
print('Setup complete!')
"
```

You should see version numbers and `Setup complete!`. If you get an error, see Troubleshooting below.

## 5. Running Code

**Important:** Always run scripts from the `cs195-ska-projects/` directory. The learning materials and project code import from `shared/`, which only resolves correctly from this root directory.

```bash
# Correct — run from the repo root
cd cs195-ska-projects
python3 my_script.py

# Wrong — imports will fail from a subdirectory
cd cs195-ska-projects/learning
python3 my_script.py    # ModuleNotFoundError: No module named 'shared'
```

If you need to run from a different directory, add the repo root to your Python path:

```bash
export PYTHONPATH="/path/to/cs195-ska-projects:$PYTHONPATH"
```

## Troubleshooting

**`ModuleNotFoundError: No module named 'shared'`**
You're running from the wrong directory. `cd` to `cs195-ska-projects/` or set `PYTHONPATH` as shown above.

**`ModuleNotFoundError: No module named 'torch'`**
Your virtual environment isn't activated. Run `source venv/bin/activate`.

**`torch.cholesky_solve` not found or deprecated**
Use `torch.linalg.solve_triangular` instead. The learning materials' Week 2 Exercise 1 hint references `torch.cholesky_solve`, but the codebase uses the modern `torch.linalg` API. Both work, but `torch.linalg.solve_triangular(L, b, upper=False)` is preferred.

**Matplotlib plots don't show up**
Add `plt.show()` at the end of your plotting code, or use `plt.savefig('plot.png')` to save to a file. In Jupyter notebooks, add `%matplotlib inline` at the top of the notebook.

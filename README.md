# Machine Learning for Autonomous Robots – Exercise Sheet 1

Group 5 solution for Exercise Sheet 1 (WiSe '26/'27, Prof. Dr. Frank Kirchner). Due 17.11.2026.

## Contents

| File | Description |
|---|---|
| `ex01.ipynb` | **Submission notebook**: solved, executed, with outputs and plots |
| `ex01 copy.ipynb` | Original blank exercise template, kept for reference |
| `SOLUTION_NOTES.md` | Detailed explanation of the solution, with links to the lecture slides |
| `main.pdf` | Preprocessing lecture slides |
| `pyproject.toml`, `uv.lock` | Python environment definition |

## Tasks

**Problem 1: Feature Selection (Iris dataset)**
- 1A: absolute Pearson feature–feature / feature–class correlations and CFS merit
- 1B: merit, `frr` and `fcr` table for all 16 feature subsets
- 1C: GreedyCFS (hill climbing). Selects features `[3, 2]`
- 1D: best subset is `[2, 3]` (petal length + width, merit ≈ 0.9618), both for cardinality 2 and overall

**Problem 2: PCA (Olivetti faces)**
- 2A: PCA from scratch (manual covariance, no `np.cov`), reconstructions with 0/5/50/200 components and eigenfaces PC1/PC2/PC10
- 2B: interpretation of the plotted component combinations

## Setup

Requires [uv](https://docs.astral.sh/uv/) and Python ≥ 3.12.

```bash
uv sync                                   # create .venv from uv.lock
.venv/bin/jupyter lab ex01.ipynb          # open the notebook
```

Re-run the whole notebook from a fresh kernel:

```bash
.venv/bin/jupyter nbconvert --to notebook --execute --inplace ex01.ipynb
```

The Olivetti faces dataset is downloaded by scikit-learn on first run.

## Submission

Upload `ex01.ipynb` via JupyterHub by 23:59 on 17.11.2026. The notebook must stay precompiled, with all outputs, and with the nbgrader cell metadata unchanged.

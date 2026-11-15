# Exercise Sheet 1 – Solution Notes

These notes explain the solution in `ex01.ipynb` (Problem 1: feature selection, Problem 2: PCA).
Slide numbers refer to the PDF pages of `main.pdf` (preprocessing lecture).

Only the `# YOUR CODE HERE` / `YOUR ANSWER HERE` placeholders were replaced, and the nbgrader
metadata is unchanged. The group-number cell is still for you to fill in.

---

# Problem 1 – Feature Selection

## 1A – Correlation and merit

### `calculate_abs_correlation(X, y)`

The class labels `y` (0, 1, 2) are stacked onto `X` as an extra column, giving a 150×5 matrix.
The function returns the **5×5 matrix of absolute Pearson correlations** between all columns:

```
            f0     f1     f2     f3    class
f0        1.000  0.118  0.872  0.818  0.783
f1        0.118  1.000  0.428  0.366  0.427
f2        0.872  0.428  1.000  0.963  0.949
f3        0.818  0.366  0.963  1.000  0.957
class     0.783  0.427  0.949  0.957  1.000
```

- `corr[i, j]` (i, j < 4) → feature-feature correlation |r_ff|
- `corr[i, 4]` (last column) → feature-class correlation |r_fc|

This is a single `ndarray`, as the docstring asks, and it has the same structure as the table on slide p. 47.

The correlation is the formula from slide p. 46, written out by hand:

$$r(f_i, f_j) = \frac{\sum_k (X_{k,i}-\bar X_i)(X_{k,j}-\bar X_j)}{\sqrt{\sum_k (X_{k,i}-\bar X_i)^2}\,\sqrt{\sum_k (X_{k,j}-\bar X_j)^2}}$$

computed for all pairs at once:

| Code                                    | Formula part                                       |
|-----------------------------------------|----------------------------------------------------|
| `centered = data - data.mean(axis=0)`   | $X_{k,i} - \bar X_i$                               |
| `centered.T @ centered`                 | numerator, $\sum_k(\dots)(\dots)$ for every (i, j) |
| `np.sqrt((centered ** 2).sum(axis=0))`  | each root in the denominator                       |
| `np.outer(norms, norms)`                | product of the two roots for every (i, j)          |

**Shorter alternative:** `np.abs(np.corrcoef(data, rowvar=False))` returns the same matrix (to about
15 decimal places). `np.corrcoef` computes covariance / (σ_i σ_j), and the 1/(n−1) factors cancel,
which leaves exactly the slide formula. `rowvar=False` tells numpy the *columns* are the variables.

### `calculate_merit(X, y, feature_set)`

Slide p. 53:

$$\text{merit}(F') = \frac{|F'|\,\overline{|r_{cf}|}}{\sqrt{|F'| + |F'|(|F'|-1)\,\overline{|r_{ff}|}}}$$

- `fcr` = mean of `corr[f, 4]` over the features in the set
- `frr` = mean of `corr[i, j]` over all **pairs** (`itertools.combinations(feature_set, 2)`)
  - no auto-correlations; for a single feature |r_ff| := 1 (slide convention)
- empty set → merit 0

For one feature the formula gives merit = fcr, which is why the `[0]`…`[3]` rows have merit = fcr.

---

## 1B – Table of all subsets

`itertools.combinations(range(4), k)` for k = 0…4 produces all 2⁴ = 16 subsets, ordered by size,
matching the sample table. For each subset the cell computes `frr` and `fcr` from the correlation
matrix, takes the merit **from the graded `calculate_merit`**, and stores one row.

Output details:
- `pd.DataFrame(rows, dtype=object)`: `object` keeps `None` for the empty set; a float column would turn it into `NaN`.
- `to_string(index=False, formatters=str)`: no 0–15 row-number column, and floats are printed in full
  (pandas would otherwise round them to 6 digits).
- Empty set and single-feature values are printed as `0.0` / `1.0` instead of the sample's `0` / `1`, because
  `calculate_merit` returns a `float` as its docstring says.

The computed values match the sample table up to floating-point rounding.

---

## 1C – GreedyCFS (slide p. 56)

Same as the slide's pseudocode:

```
F' = [], F = [0, 1, 2, 3]
repeat:
    last_merit = merit(F')
    f_sel      = argmax over f in F of merit(F' + [f])
    if merit(F' + [f_sel]) < last_merit: return F'      # strict "<", as on the slide
    move f_sel from F to F'
```

One extra stop condition was added: when `F` is empty. The slide's `while True` would otherwise
take an argmax over an empty set.

**Run on iris:**

| Step | F'      | Best candidate | Candidate merit | Merit of F' | Decision           |
|------|---------|----------------|-----------------|-------------|--------------------|
| 1    | []      | +3 → [3]       | 0.9565          | 0           | add 3              |
| 2    | [3]     | +2 → [2, 3]    | 0.9618          | 0.9565      | add 2              |
| 3    | [3, 2]  | +0 → [0, 2, 3] | 0.9328          | 0.9618      | lower → **stop**   |

Result: `The selected features are: [3, 2].` The list is in the order the features were picked,
as plain Python `int`s.

---

## 1D – Answer

- Best subset of **cardinality 2**: **[2, 3]** (petal length + petal width), merit ≈ 0.9618.
- Best subset of **any cardinality**: also **[2, 3]**, the highest merit in the whole table.
- Why: features 2 and 3 are highly redundant (|r_ff| ≈ 0.963), but both are so strongly correlated
  with the class (|r_fc| ≈ 0.953) that the pair still wins. Every third feature adds more redundancy
  than class information, so the merit drops. GreedyCFS finds the same optimum here, although
  greedy search is not guaranteed to in general.

---

# Problem 2 – Principal Component Analysis

Data: Olivetti faces, **400 images × 4096 pixels** (64×64 grayscale, scaled to [0, 255] by
`load_dataset`). sklearn downloads it once to `~/scikit_learn_data`.

## 2A – Implementation

### `pca(data, k)` – slide p. 69

```
mean   = data.mean(axis=0)                 # the "mean face", 4096 values
X      = data - mean                       # X <- X - X̄
Σ      = Xᵀ X / (n - 1)                    # 4096 × 4096 covariance, by hand (np.cov not allowed)
Λ, W   = eigh(Σ)                           # ΣW = WΛ
sort descending by eigenvalue, keep first k
return W[:, top-k].T  (k × 4096),  Λ[top-k]  (k,)
```

- **`eigh` instead of `eig`:** Σ is symmetric, so `np.linalg.eigh` is the right solver. It returns real
  values (plain `eig` can give complex numbers with tiny imaginary parts) and is faster. It sorts
  *ascending*, so the code reverses the order.
- **Rows, not columns:** numpy returns eigenvectors as the *columns* of `W`. The template treats
  `vecs[i]` as one basis vector, so the function returns them as rows: `eigen_vectors[i]` is the
  i-th principal component (an "eigenface").
- The slide's last step (projection `Y = X · W[:, :k]`) is not part of `pca` here, because the template
  asks for eigenvectors and eigenvalues. The projection is done in the main block (see below).
- The 4096×4096 eigendecomposition takes about 15 s.

### `normalized_linear_combination(vecs, weights)`

```
vec = Σ_i weights[i] · vecs[i]     for the first S = len(weights) vectors
vec = (vec - min) / (max - min)    # min-max normalisation to [0, 1]
return vec as shape (4096, 1)
```

Normalisation is needed because eigenvectors have positive *and* negative entries, so the raw
sum isn't in a pixel range. Min-max scaling just stretches it into [0, 1] for display.

### Main block – what is plotted

PCA is run with **k = 200**, enough for the largest reconstruction. Variance explained
(sum of the first k eigenvalues / total variance):

| Components | 5      | 50     | 200    |
|------------|--------|--------|--------|
| Variance   | 54.4 % | 87.4 % | 97.9 % |

**Grid 1 – reconstruction of face #0** (3 combinations + 2 reference images):

| mean face (k = 0) | k = 5 | k = 50 | k = 200 | original |
|---|---|---|---|---|

- Weights = projection of the face onto each component: $w_i = (\text{face} - \text{mean}) \cdot pc_i$
  (this is the slide's $Y = X \cdot W$ for one face).
- Reconstruction = $\text{mean} + \sum_{i<k} w_i \, pc_i$.
- To let `normalized_linear_combination` build the whole thing in one call, the **mean face is put
  in front of the eigenvectors as an extra basis vector with weight 1**:
  `basis = [mean, pc_1, …, pc_200]`, `weights = [1, w_1, …, w_k]`.
  Adding the mean *after* normalisation would mix a [0, 1] image with a [0, 255] one.
- The mean face and the original also go through the same function (weight `[1]`), so all five
  images are scaled the same way.

**Grid 2 – single eigenfaces** (3 combinations): PC1, PC2 and PC10, each made with a one-hot weight
list, e.g. PC10 = `[0]*9 + [1]`.

Plots use a small `show_grid` helper (one labelled figure per group). The template's `showVec` is
left unchanged.

## 2B – Answer (summary)

- **More components = more detail.** Components are sorted by variance. The first few capture coarse
  structure shared by many faces (lighting, outline), so k = 5 still looks like the mean face. Around
  k = 50 the person becomes recognisable, with some "ghost" artefacts from other faces. k = 200 is
  close to the original, with fine details like eyebrows and nostrils.
- **Eigenfaces:** PC1 and PC2 are smooth, global patterns (mostly illumination and face-vs-background).
  PC10 is more local: glasses, eyebrows, mouth.
- The sign of an eigenvector is arbitrary, so an eigenface may appear inverted (dark ↔ bright).

The full answer is in the notebook's 2B cell.

---

## Running it (uv)

The repo is a uv project (`pyproject.toml`, `uv.lock`; `.venv/` is gitignored).

```bash
uv sync                                                          # create .venv from uv.lock
uv run jupyter nbconvert --to notebook --execute --inplace ex01.ipynb   # re-run everything (~25 s)
```

To open the notebook in VS Code / Jupyter, pick the kernel at `.venv/bin/python`.
All saved outputs in `ex01.ipynb` came from a full top-to-bottom run in this environment.

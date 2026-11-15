# Exercise Sheet 1 – Problem 1 (Feature Selection): Solution Notes

These notes explain the Problem 1 solution in `ex01.ipynb` and `ex01 copy.ipynb`.
Slide numbers refer to the PDF pages of `main.pdf` (preprocessing lecture).

| Notebook          | 1A correlation                                   | 1B, 1C, 1D |
|-------------------|--------------------------------------------------|------------|
| `ex01.ipynb`      | **Version A** – `np.corrcoef`                    | identical  |
| `ex01 copy.ipynb` | **Version B** – Pearson formula written by hand  | identical  |

Both notebooks are complete and give the same results. Keep the one you prefer and delete the other.
Only the `# YOUR CODE HERE` / `YOUR ANSWER HERE` placeholders were replaced. The group-number
cell and Problem 2 are still for you to do.

---

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

**Version A (`ex01.ipynb`):** `np.abs(np.corrcoef(data, rowvar=False))`.
`rowvar=False` tells numpy that the *columns* are the variables (by default it uses rows).

**Version B (`ex01 copy.ipynb`):** the formula from slide p. 46,

$$r(f_i, f_j) = \frac{\sum_k (X_{k,i}-\bar X_i)(X_{k,j}-\bar X_j)}{\sqrt{\sum_k (X_{k,i}-\bar X_i)^2}\,\sqrt{\sum_k (X_{k,j}-\bar X_j)^2}}$$

computed for all pairs at once:

| Code                                    | Formula part                                   |
|-----------------------------------------|------------------------------------------------|
| `centered = data - data.mean(axis=0)`   | $X_{k,i} - \bar X_i$                           |
| `centered.T @ centered`                 | numerator, $\sum_k(\dots)(\dots)$ for every (i, j) |
| `np.sqrt((centered ** 2).sum(axis=0))`  | each root in the denominator                   |
| `np.outer(norms, norms)`                | product of the two roots for every (i, j)      |

**Why A and B agree:** `np.corrcoef` computes covariance / (σ_i σ_j). The 1/(n−1) factors in the
covariance and the two standard deviations cancel, which leaves exactly the slide formula. The
two versions differ only in the last ~15 digits (floating-point rounding).

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

## Running it (uv)

The repo is a uv project (`pyproject.toml`, `uv.lock`; `.venv/` is gitignored).

```bash
uv sync                           # create .venv from uv.lock
uv run jupyter nbconvert --help   # tools are available through uv run
```

To open the notebook in VS Code / Jupyter, pick the kernel at `.venv/bin/python`.
The saved outputs of the Problem 1 cells (1A, 1B, 1C) came from this environment.
Problem 2 was not run because it still raises `NotImplementedError`.

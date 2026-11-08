# STAT 841 / CM 763 - Assignment 2

Code and data for Questions 1, 3, 5, and 6. Author: Carlos Pineda.

## How I ran the code

I used **Anaconda3 with Python 3.12.12** in a Conda environment, run on an
Apple Silicon Mac. The recorded versions in `Q3_solver_status.txt` and
`Q6_solver_log.txt` are **NumPy 2.5.2, SciPy 1.18.0, and Matplotlib 3.11.1**.

Dependencies: **NumPy, SciPy, and Matplotlib**. (`Q1.py` and `Q5.py` need only
NumPy and Matplotlib; `Q3.py` and `Q6.py` also use `scipy.optimize.minimize`.)

## How to run

Extract the ZIP and open a terminal in the `a2` folder containing the scripts
and the `data` folder. Use an existing Python environment; activate your
existing Conda environment first if needed.

Install the dependencies in that interpreter, then run:

```bash
python -m pip install numpy scipy matplotlib
python Q1.py
python Q3.py
python Q5.py
python Q6.py
```

Each script locates its files relative to its own location, so it can be run
from any working directory. Running the scripts overwrites existing result files.

## Scripts and outputs

| Script | Purpose and outputs |
|---|---|
| `Q1.py` | RBF regression with 4 model sizes; part (b) saves `Q1_first.csv` and `Q1_first_figure.png`; part (c) runs 200 repetitions and saves `Q1_simulation.csv` and `Q1_simulation_figure.png`. |
| `Q3.py` | Fits the standard, penalized, and translated SVMs with SLSQP; saves `Q3_results.csv`, `Q3_solver_status.txt`, and `Q3_boundaries.png`. |
| `Q5.py` | Implements the supplied kernels and prediction rules and checks Gram symmetry; no model fitting. Saves `Q5_results.csv` and `Q5_boundaries.png`. |
| `Q6.py` | Fits the three soft-margin Gaussian-kernel SVMs (gamma 0.2, 1, 10) and compares with the supplied hard-margin result; saves `Q6_results.csv`, `Q6_solver_log.txt`, and `Q6_boundaries.png`. |

Keep the `data` folder beside the scripts:

- `Q3_train.csv`, `Q3_test.csv`
- `Q5_train.csv`, `Q5_test.csv`, `Q5_models.csv`, `Q5_model_info.csv`
- `Q6_train.csv`, `Q6_test.csv`, `Q6_hard_summary.csv`, `Q6_hard_grid.csv`

Observations are columns and are matched by ID.

## Reproducibility and numerical checks

- **Q1:** seed `841`; `N = 80` inputs on `[-1, 1]`, noise `sigma = 0.3`, width `h = 0.12`, model sizes `m = 5, 9, 17, 33`. Part (b) is repetition 1 of the 200 repetitions in part (c), and the same generator continues without reseeding.
- **Q3:** no randomness; SLSQP with `ftol=1e-11`, `maxiter=5000`, initial `alpha = 0`, and analytic derivatives. Feasibility and primal-dual gap checks are logged in `Q3_solver_status.txt`.
- **Q5:** no randomness and no fitting; boundaries are drawn on a 300 x 300 grid at 300 DPI.
- **Q6:** no randomness; Gaussian kernel with `h = 0.7`, SLSQP with `ftol=1e-11`, `maxiter=5000`, initial `alpha = 0`, support tolerance `1e-6`. The hard-margin row and grid scores are read unchanged from the supplied CSV files. The script raises an error if a solver or numerical check fails; see `Q6_solver_log.txt`.

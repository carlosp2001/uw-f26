# STAT 841 / CM 763 - Assignment 1

Code and data for Questions 3, 5, and 6. Author: Carlos Pineda.

## How I ran the code

I used **Anaconda3 with Python 3.12.14**, configured as the Python interpreter
in **PyCharm**. I used my existing Conda environment rather than creating a
separate environment for this assignment, and ran the scripts from PyCharm.

Dependencies: **NumPy, pandas, and Matplotlib**.

## How to run

Extract the ZIP and open a terminal in the `a1` folder containing the scripts
and CSV files. Use an existing Python environment; activate your existing Conda
environment first if needed. A new environment and PyCharm are not required.

Install the dependencies in that interpreter, then run:

```bash
python -m pip install numpy pandas matplotlib
python Q3.py
python Q5.py
python Q6.py
```

Alternatively, open the folder in PyCharm, select your existing Anaconda/Python
interpreter, install the same dependencies in it, and run each script with
this folder as the working directory.

`Q6.py` runs both computational parts. To run them separately:

```bash
python Q6.py --part b
python Q6.py --part c
```

## Scripts and outputs

| Script | Purpose and outputs |
|---|---|
| `Q3.py` | PCA versus Fisher; prints errors and saves `Q3_comparison.png`. |
| `Q5.py` | Prints probability, loss, nine gradient comparisons, and maximum discrepancy. |
| `Q6.py --part b` | Saves `Q6_cv_results.csv` and `Q6_predictions.csv`. |
| `Q6.py --part c` | Saves `Q6_early_stopping.png`, `Q6_early_stopping.csv`, and `Q6_early_stopping_summary.json`. |

Keep all four supplied datasets beside `Q6.py`: `A1_train.csv`, `A1_test.csv`,
`A1_noise_train.csv`, and `A1_resampling.csv`. Observations are columns and are
matched by ID. Running the scripts can overwrite existing result files.

## Reproducibility and numerical checks

- **Q3:** seed `42`; 20 stratified splits; NumPy linear algebra for PCA and Fisher.
- **Q5:** fixed inputs, no randomness; centered finite differences with `epsilon=1e-6`.
- **Q6(b):** supplied folds; Newton maximum 150 iterations, update/gradient tolerance `1e-8`, final maximum absolute gradient at most `1e-5`.
- **Q6(c):** supplied fit/validation split; zero initialization; 3,000 full-batch updates with step size `0.05`; losses recorded every 10 updates, including iteration 0.
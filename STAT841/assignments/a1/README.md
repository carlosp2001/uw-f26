# STAT 841 / CM 763 — Assignment 1 (Fall 2026)
Code and data for Questions 3, 5, and 6.
Environment
- Anaconda3, Python 3.12.4
- Packages: NumPy, pandas, Matplotlib
From the extracted ZIP directory:
conda env create -f environment.yml
conda activate stat841-a1
Run
python Q3.py
python Q5.py
python Q6.py
To run Question 6 separately:
python Q6.py --part b
python Q6.py --part c
The scripts use relative paths and require no external downloads. Run them in a writable directory; they may overwrite existing result files.
Included files
File	Purpose
Q3.py	Simulated PCA vs. Fisher; saves Q3_comparison.png.
Q5.py	Forward/backward passes and nine gradient checks.
Q6.py	Five-fold model selection and early stopping.
Q6_cv_results.csv	All 27 cross-validation candidates.
Q6_predictions.csv	Probabilities for 300 test observations.
environment.yml	Conda setup.


Question 6 input data (included in the ZIP):
- A1_train.csv: 600 labeled observations, 20 features.
- A1_test.csv: 300 unlabeled observations, 20 features.
- A1_noise_train.csv: 100 additional features for Q6(c).
- A1_resampling.csv: fixed CV folds and early-stopping split.
Observations are columns and are matched by ID. All preprocessing uses training data only.
Reproducibility and numerical checks
- Q3: random seed 42; 20 stratified splits; PCA and Fisher computed using NumPy linear algebra.
- Q5: fixed inputs (no seed); centered finite differences with epsilon=1e-6.
- Q6(b): supplied folds (no seed); Newton-Raphson maximum 150 iterations; update/gradient tolerance 1e-8, verified final gradient below 1e-5.
- Q6(c): supplied fit/validation split; 3,000 full-batch updates, learning rate 0.05; losses recorded every 10 updates; saves Q6_early_stopping.png.
Stable log/probability calculations and matrix solves are used. No prebuilt PCA, classification, optimization, cross-validation, automatic differentiation, or early-stopping routines are called.
Written solutions and required figures are submitted separately to Crowdmark. Rename the submission ZIP to WatIAM_A1.zip, using your Waterloo username.
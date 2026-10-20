"""STAT 841 Assignment 1 - Question 6(b) and 6(c).

Dependencies: numpy, pandas, matplotlib
Run both parts: python Q6.py
Run one part:   python Q6.py --part b   OR   python Q6.py --part c
Keep the four supplied A1_*.csv files beside this script.
Observations are columns. All model fits and update loops use basic NumPy.
Part (b) retains the previous implementation; part (c) is added below it.
"""
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ALPHAS = [0, .1, .3, .5, .7, .9]
DIMS = [2, 5, 10, 15, 20]
LAMBDAS = [.1, 1., 10., 100.]
MAX_ITER = 150
TOL = 1e-8
CLIP = 1e-15

def load():
    train = pd.read_csv(HERE / 'A1_train.csv', index_col=0)
    test = pd.read_csv(HERE / 'A1_test.csv', index_col=0)
    split = pd.read_csv(HERE / 'A1_resampling.csv', index_col=0)
    assert train.columns.is_unique and test.columns.is_unique
    assert list(train.index[1:]) == list(test.index)
    assert set(train.columns) == set(split.columns)
    split = split.loc[:, train.columns]
    return (train.iloc[1:].to_numpy(dtype=float),
            train.loc['y'].to_numpy(dtype=int),
            test.to_numpy(dtype=float),
            test.columns.to_list(),
            split.loc['fold'].to_numpy(dtype=int))

def standardize_fit(X):
    mean = X.mean(axis=1, keepdims=True)
    sd = np.sqrt(np.mean((X - mean)**2, axis=1, keepdims=True))
    if np.any(sd < 1e-12):
        raise ValueError('Feature with near-zero training standard deviation')
    return mean, sd

def apply_standardize(X, mean, sd):
    return (X - mean) / sd

def pca_fit(X, r):
    mean = X.mean(axis=1, keepdims=True)
    centered = X - mean
    eigvals, eigvecs = np.linalg.eigh(centered @ centered.T / X.shape[1])
    return mean, eigvecs[:, -r:][:, ::-1]

def pca_apply(X, mean, basis):
    return basis.T @ (X - mean)

def class_stats(X, y):
    priors, means, covs = [], [], []
    for k in (0, 1):
        group = X[:, y == k]
        mu = group.mean(axis=1)
        residual = group - mu[:, None]
        priors.append(group.shape[1] / X.shape[1])
        means.append(mu)
        covs.append(residual @ residual.T / group.shape[1])
    return np.array(priors), means, covs

def gaussian_log_density(X, mu, cov):
    d = X.shape[0]
    sign, logdet = np.linalg.slogdet(cov)
    if sign <= 0:
        raise ValueError('Covariance is not positive definite')
    residual = X - mu[:, None]
    quadratic = np.sum(residual * np.linalg.solve(cov, residual), axis=0)
    return -.5 * (d * np.log(2 * np.pi) + logdet + quadratic)

def generative_predict(X_train, y_train, X_val, family, alpha=None):
    priors, means, covs = class_stats(X_train, y_train)
    if family == 'LDA':
        pooled = priors[0] * covs[0] + priors[1] * covs[1]
        covs = [pooled, pooled]
    else:
        d = X_train.shape[0]
        covs = [(1-alpha)*C + alpha*np.trace(C)/d*np.eye(d) for C in covs]
    logits = np.vstack([np.log(priors[k]) + gaussian_log_density(X_val, means[k], covs[k]) for k in (0, 1)])
    # stable two-class normalization
    return np.exp(logits[1] - np.logaddexp(logits[0], logits[1]))

def sigmoid(z):
    return np.exp(-np.logaddexp(0., -z))

def logistic_objective(A, y, params, lam):
    z = A @ params
    return float(np.sum(np.logaddexp(0., z) - y*z) + .5*lam*np.dot(params[1:], params[1:]))

def logistic_fit(X, y, lam):
    # Features are columns; intercept is not penalized.
    A = np.column_stack((np.ones(X.shape[1]), X.T))
    params = np.zeros(A.shape[1])
    penalty = np.r_[0., np.full(X.shape[0], lam)]
    converged = False
    for iteration in range(1, MAX_ITER + 1):
        z = A @ params
        p = sigmoid(z)
        grad = A.T @ (p-y) + penalty*params
        w = p*(1-p)
        H = A.T @ (w[:, None]*A) + np.diag(penalty)
        delta = np.linalg.solve(H, grad)
        # Backtracking prevents increases in penalized negative log likelihood.
        step = 1.
        old_obj = logistic_objective(A, y, params, lam)
        while step > 2**-30:
            candidate = params - step*delta
            if logistic_objective(A, y, candidate, lam) <= old_obj + 1e-12:
                break
            step *= .5
        if step <= 2**-30:
            raise RuntimeError('Newton line search failed')
        params = candidate
        if np.max(np.abs(step*delta)) < TOL or np.max(np.abs(grad)) < TOL:
            converged = True
            break
    if not converged:
        raise RuntimeError(f'Logistic regression did not converge within {MAX_ITER} iterations')
    final_grad = A.T @ (sigmoid(A @ params)-y) + penalty*params
    if np.max(np.abs(final_grad)) > 1e-5:
        raise RuntimeError('Logistic gradient convergence check failed')
    return params, iteration, float(np.max(np.abs(final_grad)))

def logistic_predict(X, params):
    return sigmoid(params[0] + params[1:] @ X)

def metrics(y, p):
    error = np.mean((p >= .5) != y)
    # p is clipped only for log loss, not for classification.
    p = np.clip(p, CLIP, 1-CLIP)
    loss = -np.mean(y*np.log(p) + (1-y)*np.log1p(-p))
    return float(error), float(loss)

def candidates():
    yield ('LDA', None, None, None)
    for a in ALPHAS:
        yield ('QDA', a, None, None)
    for r in DIMS:
        for lam in LAMBDAS:
            yield ('PCA-logistic', None, r, lam)

def fit_predict(Xtr, ytr, Xva, candidate):
    model, alpha, r, lam = candidate
    if model in ('LDA', 'QDA'):
        return generative_predict(Xtr, ytr, Xva, model, alpha), None
    mean, basis = pca_fit(Xtr, r)
    train_scores = pca_apply(Xtr, mean, basis)
    val_scores = pca_apply(Xva, mean, basis)
    params, iters, grad = logistic_fit(train_scores, ytr, lam)
    return logistic_predict(val_scores, params), (iters, grad)

def selection_key(row):
    model = row['model']
    family = {'LDA': 0, 'QDA': 1, 'PCA-logistic': 2}[model]
    return (row['mean_cv_log_loss'], family,
            -row['alpha'] if model == 'QDA' else 0,
            row['r'] if model == 'PCA-logistic' else 0,
            -row['lambda'] if model == 'PCA-logistic' else 0)

def run_part_b():
    X, y, X_test, test_ids, folds = load()
    assert X.shape == (20, 600) and X_test.shape == (20, 300)
    assert set(folds) == {1,2,3,4,5}
    results = []
    convergence = []
    for candidate in candidates():
        model, alpha, r, lam = candidate
        errors, losses = [], []
        for fold in range(1, 6):
            tr, va = folds != fold, folds == fold
            mu, sd = standardize_fit(X[:, tr])
            Xtr = apply_standardize(X[:, tr], mu, sd)
            Xva = apply_standardize(X[:, va], mu, sd)
            probabilities, check = fit_predict(Xtr, y[tr], Xva, candidate)
            err, loss = metrics(y[va], probabilities)
            errors.append(err)
            losses.append(loss)
            if check is not None:
                convergence.append((candidate, fold, *check))
        results.append(dict(model=model, alpha=alpha, r=r, **{'lambda':lam},
                            mean_cv_error=np.mean(errors), se_cv_error=np.std(errors, ddof=1)/np.sqrt(5),
                            mean_cv_log_loss=np.mean(losses), se_cv_log_loss=np.std(losses, ddof=1)/np.sqrt(5)))
    assert len(results) == 27
    df = pd.DataFrame(results)
    df.to_csv(HERE / 'Q6_cv_results.csv', index=False, float_format='%.17g')
    selected = min(results, key=selection_key)
    print('Candidate models:', len(results))
    for family in ['LDA', 'QDA', 'PCA-logistic']:
        best = min((row for row in results if row['model'] == family), key=selection_key)
        print('Family best:', best)
    print('Overall selected:', selected)
    print('Logistic fold fits:', len(convergence), 'max iterations:', max(t[2] for t in convergence),
          'max abs final gradient:', max(t[3] for t in convergence))
    mu, sd = standardize_fit(X)
    Xtr, Xte = apply_standardize(X, mu, sd), apply_standardize(X_test, mu, sd)
    choice = (selected['model'], selected['alpha'], selected['r'], selected['lambda'])
    predictions, check = fit_predict(Xtr, y, Xte, choice)
    if check: print('Refit convergence: iterations, max abs gradient:', check)
    assert len(predictions) == 300 and np.all(np.isfinite(predictions)) and np.all((0 <= predictions) & (predictions <= 1))
    with (HERE / 'Q6_predictions.csv').open('w') as file:
        file.write('observation,' + ','.join(test_ids) + '\n')
        file.write('p1,' + ','.join(format(float(p), '.17g') for p in predictions) + '\n')
    print('Wrote Q6_cv_results.csv and Q6_predictions.csv')


# -------------------------------------------------------------------------
# Question 6(c): early stopping on average, unpenalized negative log likelihood.
# This experiment is independent of the model selected in part (b).
# -------------------------------------------------------------------------
EARLY_UPDATES = 3000
EARLY_STEP_SIZE = 0.05
EARLY_RECORD_EVERY = 10


def load_early_stopping_data():
    """Align original features, noise features, and split membership by ID."""
    train = pd.read_csv(HERE / 'A1_train.csv', index_col=0)
    noise = pd.read_csv(HERE / 'A1_noise_train.csv', index_col=0)
    resampling = pd.read_csv(HERE / 'A1_resampling.csv', index_col=0)

    for name, frame in [('training', train), ('noise', noise),
                        ('resampling', resampling)]:
        if not frame.columns.is_unique or not frame.index.is_unique:
            raise ValueError(f'Duplicate observation IDs or row names in {name}')
    ids = train.columns
    if len(ids) != 600:
        raise ValueError('Expected 600 labeled observations')
    if set(noise.columns) != set(ids) or set(resampling.columns) != set(ids):
        raise ValueError('Observation IDs do not agree across input files')

    # Reorder by the original training IDs, not by the files' column positions.
    noise = noise.loc[:, ids]
    resampling = resampling.loc[:, ids]
    original_rows = [f'x{i}' for i in range(1, 21)]
    noise_rows = [f'z{i}' for i in range(1, 101)]
    if set(train.index) != {'y', *original_rows}:
        raise ValueError('Training rows must be y and x1 through x20')
    if set(noise.index) != set(noise_rows):
        raise ValueError('Noise rows must be z1 through z100')
    if 'early_stop' not in resampling.index:
        raise ValueError('Missing early_stop membership row')

    X = np.vstack((train.loc[original_rows].to_numpy(dtype=float),
                   noise.loc[noise_rows].to_numpy(dtype=float)))
    y = train.loc['y'].to_numpy(dtype=float)
    if X.shape != (120, 600) or not np.all(np.isfinite(X)):
        raise ValueError('Expected a finite 120 by 600 feature matrix')
    if not np.all(np.isfinite(y)) or not np.all(np.isin(y, [0, 1])):
        raise ValueError('Labels must be 0 or 1')

    membership = resampling.loc['early_stop'].to_numpy(dtype=str)
    if set(membership) != {'fit', 'validation'}:
        raise ValueError('early_stop must contain fit and validation')
    fit_mask = membership == 'fit'
    val_mask = membership == 'validation'
    if not fit_mask.any() or not val_mask.any():
        raise ValueError('Both fit and validation subsets must be nonempty')
    return X, y, fit_mask, val_mask


def average_log_loss_from_logits(y, logits):
    """Stable binary log loss. No probability clipping or L2 penalty."""
    # y=1: log(1 + exp(-logit)); y=0: log(1 + exp(logit)).
    # logaddexp avoids overflow without changing the mathematical objective.
    signed_logits = (1.0 - 2.0 * y) * logits
    return float(np.mean(np.logaddexp(0.0, signed_logits)))


def early_stopping_fit(X_fit, y_fit, X_val, y_val):
    """Run every prescribed update; select only among recorded checkpoints."""
    intercept = 0.0
    beta = np.zeros(X_fit.shape[0], dtype=float)
    n_fit = X_fit.shape[1]
    history = []
    best_loss = np.inf
    best_iteration = None
    best_intercept = None
    best_beta = None

    # Iteration t means exactly t completed, simultaneous parameter updates.
    for iteration in range(EARLY_UPDATES + 1):
        fit_logits = intercept + beta @ X_fit

        if iteration % EARLY_RECORD_EVERY == 0:
            val_logits = intercept + beta @ X_val
            fit_loss = average_log_loss_from_logits(y_fit, fit_logits)
            val_loss = average_log_loss_from_logits(y_val, val_logits)
            if not np.isfinite(fit_loss) or not np.isfinite(val_loss):
                raise FloatingPointError('Non-finite loss during gradient descent')
            history.append({
                'iteration': iteration,
                'fit_loss': fit_loss,
                'validation_loss': val_loss,
                'coefficient_norm': float(np.linalg.norm(beta)),
            })
            # A strict improvement preserves the earliest checkpoint on a tie.
            if val_loss < best_loss:
                best_loss = val_loss
                best_iteration = iteration
                best_intercept = intercept
                best_beta = beta.copy()

        if iteration == EARLY_UPDATES:
            break

        # Both derivatives use the SAME current intercept and coefficient vector.
        residual = sigmoid(fit_logits) - y_fit
        grad_intercept = float(np.mean(residual))
        grad_beta = (X_fit @ residual) / n_fit

        # Full-batch gradient descent, no PCA, no L2 penalty, no line search.
        intercept, beta = (
            intercept - EARLY_STEP_SIZE * grad_intercept,
            beta - EARLY_STEP_SIZE * grad_beta,
        )
        if not np.isfinite(intercept) or not np.all(np.isfinite(beta)):
            raise FloatingPointError('Non-finite parameters during gradient descent')

    expected_records = EARLY_UPDATES // EARLY_RECORD_EVERY + 1
    if len(history) != expected_records:
        raise RuntimeError('Incorrect number of recorded checkpoints')
    return history, best_iteration, best_intercept, best_beta


def save_early_stopping_figure(history, best_iteration, path):
    """Save the requested comparison figure; no GUI backend is needed."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    iterations = np.array([row['iteration'] for row in history])
    fit_losses = np.array([row['fit_loss'] for row in history])
    val_losses = np.array([row['validation_loss'] for row in history])
    best_index = int(np.flatnonzero(iterations == best_iteration)[0])

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(iterations, fit_losses, linewidth=2, label='Fit loss')
    ax.plot(iterations, val_losses, linewidth=2, label='Validation loss')
    ax.axvline(best_iteration, linestyle='--', linewidth=1.2,
               label=f'Selected iteration: {best_iteration}')
    ax.scatter([best_iteration], [val_losses[best_index]],
               marker='o', s=65, zorder=4)
    ax.set(xlabel='Completed gradient-descent updates',
           ylabel='Average negative log likelihood (log loss)',
           title='Question 6(c): Early stopping with 120 features',
           xlim=(0, EARLY_UPDATES))
    ax.grid(alpha=0.25)
    ax.legend(loc='best')
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def run_part_c():
    """Run the fixed fit/validation experiment without reading test features."""
    import json

    X, y, fit_mask, val_mask = load_early_stopping_data()
    # Fit preprocessing only on the fit subset; reuse it unchanged for validation.
    mean, sd = standardize_fit(X[:, fit_mask])
    X_fit = apply_standardize(X[:, fit_mask], mean, sd)
    X_val = apply_standardize(X[:, val_mask], mean, sd)
    y_fit, y_val = y[fit_mask], y[val_mask]

    history, selected_iteration, intercept, beta = early_stopping_fit(
        X_fit, y_fit, X_val, y_val
    )
    history_frame = pd.DataFrame(history)
    history_frame.to_csv(HERE / 'Q6_early_stopping.csv', index=False,
                         float_format='%.17g')
    selected = next(row for row in history
                    if row['iteration'] == selected_iteration)
    summary = {
        'features': int(X.shape[0]),
        'fit_observations': int(fit_mask.sum()),
        'validation_observations': int(val_mask.sum()),
        'learning_rate': EARLY_STEP_SIZE,
        'completed_updates': EARLY_UPDATES,
        'record_every': EARLY_RECORD_EVERY,
        'recorded_checkpoints': len(history),
        'selected_iteration': selected_iteration,
        'initial_fit_loss': history[0]['fit_loss'],
        'initial_validation_loss': history[0]['validation_loss'],
        'selected_fit_loss': selected['fit_loss'],
        'selected_validation_loss': selected['validation_loss'],
        'final_fit_loss': history[-1]['fit_loss'],
        'final_validation_loss': history[-1]['validation_loss'],
        'selected_coefficient_norm': selected['coefficient_norm'],
        'final_coefficient_norm': history[-1]['coefficient_norm'],
        'selected_intercept': float(intercept),
    }
    with (HERE / 'Q6_early_stopping_summary.json').open('w') as file:
        json.dump(summary, file, indent=2, allow_nan=False)
        file.write('\n')

    figure = HERE / 'Q6_early_stopping.png'
    save_early_stopping_figure(history, selected_iteration, figure)

    print('\nQuestion 6(c): Early stopping')
    print(f'Features: {X.shape[0]} (20 original + 100 noise)')
    print(f'Fit observations: {fit_mask.sum()}; '
          f'validation observations: {val_mask.sum()}')
    print('Initialization: all coefficients and intercept = 0')
    print(f'Full-batch updates: {EARLY_UPDATES}; step size: {EARLY_STEP_SIZE}')
    print(f'Recorded checkpoints: {len(history)} '
          f'(0, {EARLY_RECORD_EVERY}, ..., {EARLY_UPDATES})')
    print(f'Selected iteration (earliest minimum): {selected_iteration}')
    print('Checkpoint        Fit loss          Validation loss')
    for row in [history[0], selected, history[-1]]:
        print(f"{row['iteration']:<17d} {row['fit_loss']:.12f}    "
              f"{row['validation_loss']:.12f}")
    print('Wrote Q6_early_stopping.csv, Q6_early_stopping_summary.json, '
          'and Q6_early_stopping.png')
    return summary


def main():
    import argparse
    parser = argparse.ArgumentParser(description='STAT 841 A1 Question 6(b,c)')
    parser.add_argument('--part', choices=['b', 'c', 'all'], default='all',
                        help='Part to run (default: both b and c)')
    args = parser.parse_args()
    if args.part in ('b', 'all'):
        run_part_b()
    if args.part in ('c', 'all'):
        run_part_c()


if __name__ == '__main__':
    main()

"""STAT 841 / CM 763: Q3(b), with numerical checks for the fitted models."""

import csv
from pathlib import Path
import sys

import numpy as np
import scipy
from scipy.optimize import minimize

SUPPORT_TOL = 1e-6
CHECK_TOL = 1e-5
FTOL = 1e-11
MAXITER = 5000


def load_data(filename: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Read named rows: X has shape (2, n), and y has shape (n,)."""
    with Path(filename).open(newline="", encoding="utf-8-sig") as file:
        table = list(csv.reader(file))
    rows = {row[0]: np.array(row[1:], dtype=float) for row in table[1:]}
    if not all(name in rows for name in ("x1", "x2", "y")):
        raise ValueError("The CSV must contain rows named x1, x2, and y.")
    X = np.vstack((rows["x1"], rows["x2"]))
    y = rows["y"]
    if X.shape[1] != y.size or y.size != len(table[0]) - 1:
        raise ValueError("Features, labels, and observation IDs must align.")
    if not np.all(np.isfinite(X)) or not np.all(np.isin(y, [-1, 1])):
        raise ValueError("Features must be finite and labels must be -1 or +1.")
    return X, y


def fit_svm(Z: np.ndarray, y: np.ndarray, rho: float = 0.0) -> dict:
    """Fit the prescribed dual using columns as observations: Z is (d, n)."""
    Z = np.asarray(Z, dtype=float)
    y = np.asarray(y, dtype=float)
    if Z.ndim != 2 or y.ndim != 1 or Z.shape[1] != y.size:
        raise ValueError("Expected Z of shape (d, n) and y of shape (n,).")
    if rho < 0 or not np.all(np.isin(y, [-1, 1])):
        raise ValueError("rho must be nonnegative and labels must be -1 or +1.")

    n = y.size
    G = np.outer(y, y) * (Z.T @ Z)

    def objective(alpha):
        # SLSQP minimizes, so use the negative dual objective.
        value = 0.5 * alpha @ G @ alpha - np.sum(alpha)
        if rho > 0:
            value += (y @ alpha) ** 2 / (2 * rho)
        return float(value)

    def gradient(alpha):
        value = G @ alpha - np.ones(n)
        if rho > 0:
            value += y * (y @ alpha) / rho
        return value

    # Only the unpenalized intercept imposes y.T @ alpha = 0.
    constraints = []
    if rho == 0:
        constraints = [{
            "type": "eq",
            "fun": lambda alpha: y @ alpha,
            "jac": lambda alpha: y,
        }]

    solver = minimize(
        objective,
        np.zeros(n),
        method="SLSQP",
        jac=gradient,
        bounds=[(0.0, None)] * n,
        constraints=constraints,
        options={"ftol": FTOL, "maxiter": MAXITER},
    )

    # Keep the computed multipliers unchanged.
    alpha = solver.x
    beta = Z @ (alpha * y)
    support = alpha > SUPPORT_TOL
    if rho == 0:
        if not np.any(support):
            raise RuntimeError("No numerical support vectors; check the solver.")
        beta0 = float(np.mean(y[support] - beta @ Z[:, support]))
    else:
        beta0 = float(y @ alpha / rho)

    return {
        "beta": beta, "beta0": beta0, "alpha": alpha,
        "support": support, "solver": solver,
    }


def predict(Z: np.ndarray, beta: np.ndarray, beta0: float) -> np.ndarray:
    """Predict for any number of columns; a zero score predicts +1."""
    Z = np.asarray(Z, dtype=float)
    beta = np.asarray(beta, dtype=float)
    if Z.ndim != 2 or beta.ndim != 1 or Z.shape[0] != beta.size:
        raise ValueError("Expected Z of shape (d, n) and beta of shape (d,).")
    scores = beta @ Z + beta0
    return np.where(scores >= 0, 1, -1)


def solution_checks(Z: np.ndarray, y: np.ndarray, model: dict, rho: float):
    """Calculate the assignment's checks using training data only."""
    beta, beta0, alpha = model["beta"], model["beta0"], model["alpha"]
    G = np.outer(y, y) * (Z.T @ Z)
    scores = beta @ Z + beta0

    primal = 0.5 * beta @ beta + 0.5 * rho * beta0 ** 2
    dual = np.sum(alpha) - 0.5 * alpha @ G @ alpha
    if rho > 0:
        dual -= (y @ alpha) ** 2 / (2 * rho)

    v_feas = max(0.0, -float(np.min(alpha)), float(np.max(1 - y * scores)))
    if rho == 0:
        v_feas = max(v_feas, abs(float(y @ alpha)))
    v_gap = abs(primal - dual) / max(1.0, abs(primal), abs(dual))
    return float(primal), float(dual), float(v_feas), float(v_gap)


def main():
    folder = Path(__file__).resolve().parent
    X_train, y_train = load_data(folder / "data/Q3_train.csv")
    X_test, y_test = load_data(folder / "data/Q3_test.csv")

    configurations = [
        ("standard", 0.0, np.zeros(2)),
        ("penalized", 50.0, np.zeros(2)),
        ("translated", 50.0, np.array([-2.0, 2.0])),
    ]
    rows = []
    status_lines = [
        f"Python {sys.version.split()[0]}; NumPy {np.__version__}; SciPy {scipy.__version__}",
        f"SLSQP; ftol={FTOL}; maxiter={MAXITER}; initial alpha=zeros(n)",
        "Analytic objective and constraint derivatives; no randomness.",
    ]

    for name, rho, shift in configurations:
        # Do not standardize. Translate both datasets for the translated model.
        Z_train = X_train - shift[:, None]
        Z_test = X_test - shift[:, None]
        model = fit_svm(Z_train, y_train, rho)
        primal, dual, v_feas, v_gap = solution_checks(Z_train, y_train, model, rho)

        beta, beta0 = model["beta"], model["beta0"]
        beta0_original = float(beta0 - beta @ shift)
        margin = float(1 / np.linalg.norm(beta))
        support_count = int(np.sum(model["support"]))

        # Evaluate the test data only after fitting; never use them to tune.
        train_error = float(np.mean(predict(Z_train, beta, beta0) != y_train))
        test_error = float(np.mean(predict(Z_test, beta, beta0) != y_test))
        rows.append([
            name, beta[0], beta[1], beta0, beta0_original, margin,
            support_count, train_error, test_error, v_feas, v_gap,
        ])

        solver = model["solver"]
        checked = v_feas <= CHECK_TOL and v_gap <= CHECK_TOL
        message = (
            f"{name}: success={solver.success}, status={solver.status}, "
            f"iterations={solver.nit}, message={solver.message}\n"
            f"  primal={primal:.17g}, dual={dual:.17g}\n"
            f"  v_feas={v_feas:.17g}, v_gap={v_gap:.17g}, checks_passed={checked}"
        )
        status_lines.append(message)
        print(message)
        print(f"  beta={beta}, beta0_fit={beta0:.10f}, beta0_original={beta0_original:.10f}")
        print(f"  margin={margin:.10f}, support={support_count}, train_error={train_error}, test_error={test_error}")
        if not solver.success or not checked:
            print("  WARNING: investigate the solver status or numerical checks.")

    header = [
        "model", "beta1", "beta2", "beta0_fit", "beta0_original", "margin",
        "support_count", "train_error", "test_error", "v_feas", "v_gap",
    ]
    with (folder / "Q3_results.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(header)
        for row in rows:
            # Preserve integer counts and save floats with 17 significant digits.
            writer.writerow([format(value, ".17g") if isinstance(value, (float, np.floating)) else value for value in row])
    (folder / "Q3_solver_status.txt").write_text("\n\n".join(status_lines) + "\n", encoding="utf-8")
    print("Saved Q3_results.csv and Q3_solver_status.txt next to Q3.py.")


if __name__ == "__main__":
    main()

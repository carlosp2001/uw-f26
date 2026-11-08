"""STAT 841 / CM 763: Q6(c)-(d), fits, checks, and boundary comparison.

Only the three prescribed soft-margin models are fitted. The hard-margin
summary and grid scores are read unchanged from the supplied CSV files.
"""

import csv
from pathlib import Path
import sys

import numpy as np
import scipy
import matplotlib
from scipy.optimize import minimize

WIDTH = 0.7
GAMMAS = (0.2, 1.0, 10.0)
SUPPORT_TOL = 1e-6
CHECK_TOL = 1e-5
FTOL = 1e-11
MAXITER = 5000
ALTERED_IDS = ("train004", "train009", "train024", "train029")
RESULT_COLUMNS = (
    "model", "gamma", "train_error", "test_error", "support_count",
    "free_count", "beta0", "slack_sum", "v_feas", "v_gap",
)


def load_data(filename: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Read the supplied layout: X is (2, n), y is (n,)."""
    with Path(filename).open(newline="", encoding="utf-8-sig") as handle:
        table = list(csv.reader(handle))
    ids = table[0][1:]
    rows = {row[0]: np.asarray(row[1:], dtype=float) for row in table[1:]}
    if not all(name in rows for name in ("x1", "x2", "y")):
        raise ValueError("Input must contain rows x1, x2, and y.")
    X = np.vstack((rows["x1"], rows["x2"]))
    y = rows["y"]
    if X.shape[1] != y.size or y.size != len(ids):
        raise ValueError("Feature columns, labels, and IDs must align.")
    if not np.all(np.isfinite(X)) or not np.all(np.isin(y, [-1, 1])):
        raise ValueError("Features must be finite and labels must be -1 or +1.")
    # Keep observation order and the already altered labels unchanged.
    return X, y


def gaussian_kernel(X: np.ndarray, Z: np.ndarray) -> np.ndarray:
    """Return k(x_i, z_j) with shape (n_train, n_new), using h=0.7."""
    X = np.asarray(X, dtype=float)
    Z = np.asarray(Z, dtype=float)
    if X.ndim != 2 or Z.ndim != 2 or X.shape[0] != Z.shape[0]:
        raise ValueError("X and Z must have the same number of feature rows.")
    differences = X[:, :, None] - Z[:, None, :]
    squared_distances = np.sum(differences**2, axis=0)
    # No factor of 2 and no added diagonal penalty.
    return np.exp(-squared_distances / WIDTH**2)


def fit_soft_margin(X: np.ndarray, y: np.ndarray, gamma: float) -> dict:
    """Construct and solve the prescribed dual using training data only."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    if X.ndim != 2 or y.ndim != 1 or X.shape[1] != y.size:
        raise ValueError("Expected X of shape (d, n) and y of shape (n,).")
    if gamma <= 0 or not np.isfinite(gamma):
        raise ValueError("gamma must be positive and finite.")
    if not np.all(np.isin(y, [-1, 1])) or not np.all(np.isfinite(X)):
        raise ValueError("Invalid feature values or labels.")

    n = y.size
    K = gaussian_kernel(X, X)
    G = np.outer(y, y) * K

    def objective(alpha):
        # Minimize the negative dual objective.
        return float(0.5 * alpha @ G @ alpha - np.sum(alpha))

    def gradient(alpha):
        return G @ alpha - np.ones(n)

    equality = {
        "type": "eq",
        "fun": lambda alpha: float(y @ alpha),
        "jac": lambda alpha: y,
    }
    solver = minimize(
        objective,
        np.zeros(n),
        method="SLSQP",
        jac=gradient,
        bounds=[(0.0, gamma)] * n,
        constraints=[equality],
        options={"ftol": FTOL, "maxiter": MAXITER},
    )

    # Use all multipliers exactly as returned; do not clip or threshold them.
    alpha = solver.x
    support = alpha > SUPPORT_TOL
    free = (alpha > SUPPORT_TOL) & (alpha < gamma - SUPPORT_TOL)
    if not np.any(free):
        raise RuntimeError(
            f"gamma={gamma}: no free support vectors. "
            f"Check the solver and implementation: {solver.message}"
        )

    signed_alpha = alpha * y
    scores_without_intercept = K.T @ signed_alpha
    beta0 = float(np.mean(y[free] - scores_without_intercept[free]))
    train_scores = scores_without_intercept + beta0
    slacks = np.maximum(1.0 - y * train_scores, 0.0)

    # Evaluate the primal, dual, and both prescribed numerical checks.
    primal = float(0.5 * signed_alpha @ K @ signed_alpha + gamma * np.sum(slacks))
    dual = float(np.sum(alpha) - 0.5 * alpha @ G @ alpha)
    v_feas = max(
        0.0, -float(np.min(alpha)),
        float(np.max(alpha)) - gamma, abs(float(y @ alpha)),
        )
    v_gap = abs(primal - dual) / max(1.0, abs(primal), abs(dual))

    return {
        "gamma": gamma, "alpha": alpha, "beta0": beta0,
        "support": support, "free": free, "slacks": slacks,
        "train_scores": train_scores, "primal": primal, "dual": dual,
        "v_feas": v_feas, "v_gap": v_gap, "solver": solver,
    }


def predict(X_train: np.ndarray, y_train: np.ndarray,
            model: dict, X_new: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return scores and labels for any compatible number of new columns."""
    C = gaussian_kernel(X_train, X_new)
    scores = C.T @ (model["alpha"] * y_train) + model["beta0"]
    labels = np.where(scores >= 0.0, 1, -1)
    return scores, labels


def hard_reference_row(filename: str | Path) -> dict:
    """Copy the supplied hard-margin summary without fitting a hard model."""
    with Path(filename).open(newline="", encoding="utf-8-sig") as handle:
        summaries = list(csv.DictReader(handle))
    if len(summaries) != 1 or summaries[0]["model"] != "hard":
        raise ValueError("Expected one supplied hard-margin reference row.")
    summary = summaries[0]
    if float(summary["h"]) != WIDTH:
        raise ValueError("The hard-margin reference must use h=0.7.")
    row = {name: "" for name in RESULT_COLUMNS}
    row.update({
        "model": "hard",
        "train_error": float(summary["train_error"]),
        "test_error": float(summary["test_error"]),
        "support_count": int(summary["support_count"]),
    })
    return row


def load_grid(filename: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Read the supplied 80 by 80 grid, keeping its original column order."""
    with Path(filename).open(newline="", encoding="utf-8-sig") as handle:
        table = list(csv.reader(handle))
    rows = {row[0]: np.asarray(row[1:], dtype=float) for row in table[1:]}
    X_grid = np.vstack((rows["x1"], rows["x2"]))
    hard_scores = rows["score"]
    if X_grid.shape != (2, 6400) or hard_scores.shape != (6400,):
        raise ValueError("Expected the supplied 80 by 80 grid (6400 columns).")
    if not np.all(np.isfinite(X_grid)) or not np.all(np.isfinite(hard_scores)):
        raise ValueError("Grid coordinates and scores must be finite.")
    return X_grid, hard_scores


def plot_boundaries(X_train: np.ndarray, y_train: np.ndarray,
                    train_ids: list[str], models: list[dict],
                    X_grid: np.ndarray, hard_scores: np.ndarray,
                    filename: str | Path) -> None:
    """Draw four boundaries on the same supplied grid, without changing labels."""
    # A local canvas saves the PNG without changing IPython's plot backend.
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.lines import Line2D

    if len(train_ids) != X_train.shape[1]:
        raise ValueError("Training IDs must match the observation columns.")
    if not set(ALTERED_IDS).issubset(train_ids):
        raise ValueError("The four altered-label IDs must be present.")
    altered = np.array([obs_id in ALTERED_IDS for obs_id in train_ids])

    # Reshape the supplied coordinates; x1 varies fastest in the CSV.
    grid_x1 = X_grid[0].reshape(80, 80)
    grid_x2 = X_grid[1].reshape(80, 80)
    if not np.allclose(grid_x1, grid_x1[0:1, :]):
        raise ValueError("Unexpected x1 grid ordering.")
    if not np.allclose(grid_x2, grid_x2[:, 0:1]):
        raise ValueError("Unexpected x2 grid ordering.")

    # Use all computed coefficients for the soft-margin grid scores.
    grid_scores = [predict(X_train, y_train, model, X_grid)[0] for model in models]
    grid_scores.append(hard_scores)  # The hard scores are supplied, not fitted.
    titles = [f"Soft margin ($\\gamma = {model['gamma']:g}$)" for model in models]
    titles.append("Hard margin (supplied)")

    fig = Figure(figsize=(9.2, 9.0))
    FigureCanvasAgg(fig)
    axes = fig.subplots(2, 2)
    colors = matplotlib.rcParams["axes.prop_cycle"].by_key()["color"]
    handles = None
    for ax, title, scores in zip(axes.flat, titles, grid_scores):
        negative = y_train == -1
        positive = y_train == 1
        minus = ax.scatter(X_train[0, negative], X_train[1, negative],
                           s=32, marker="o", color=colors[0], zorder=3,
                           label="Supplied label -1")
        plus = ax.scatter(X_train[0, positive], X_train[1, positive],
                          s=40, marker="^", color=colors[1], zorder=3,
                          label="Supplied label +1")
        rings = ax.scatter(X_train[0, altered], X_train[1, altered],
                           s=145, facecolors="none", edgecolors="black",
                           linewidths=1.4, zorder=4, label="Reversed-label observation")
        boundary = ax.contour(grid_x1, grid_x2, scores.reshape(80, 80),
                              levels=[0.0], colors="black", linewidths=1.6)
        ax.clabel(boundary, fmt={0.0: "$g=0$"}, fontsize=9, inline=True)
        ax.set_title(title, fontsize=12, pad=10)
        ax.set_xlabel("$x_1$", fontsize=12)
        ax.set_ylabel("$x_2$", fontsize=12)
        ax.set_xlim(-2.2, 2.2)
        ax.set_ylim(-2.2, 2.2)
        ax.set_aspect("equal")
        ax.set_xticks([-2, -1, 0, 1, 2])
        ax.set_yticks([-2, -1, 0, 1, 2])
        handles = [minus, plus, rings,
                   Line2D([0], [0], color="black", lw=1.6, label="Decision boundary $g=0$")]

    fig.subplots_adjust(left=0.08, right=0.98, top=0.95, bottom=0.14,
                        hspace=0.28, wspace=0.24)
    fig.legend(handles=handles, loc="lower center", ncol=2,
               bbox_to_anchor=(0.53, 0.015), fontsize=10, frameon=True)
    fig.savefig(filename, dpi=300, bbox_inches="tight")
    fig.clear()


def main() -> None:
    folder = Path(__file__).resolve().parent
    X_train, y_train = load_data(folder / "data/Q6_train.csv")

    # Fit every prescribed penalty before reading test labels.
    models = [fit_soft_margin(X_train, y_train, gamma) for gamma in GAMMAS]
    X_test, y_test = load_data(folder / "data/Q6_test.csv")

    log = [
        "STAT 841 / CM 763 - Q6(c)-(d)",
        f"Python {sys.version.split()[0]}; NumPy {np.__version__}; SciPy {scipy.__version__}; Matplotlib {matplotlib.__version__}",
        f"Gaussian h={WIDTH}; gamma={GAMMAS}; support tolerance={SUPPORT_TOL}",
        f"SLSQP; analytic derivatives; initial alpha=zeros(n); ftol={FTOL}; maxiter={MAXITER}",
        "No random seed: all data are supplied and initialization is deterministic.",
        "Training labels and features are used unchanged. Test data are for evaluation only.",
        "",
    ]
    rows = []
    failed = False
    for model in models:
        _, train_labels = predict(X_train, y_train, model, X_train)
        _, test_labels = predict(X_train, y_train, model, X_test)
        row = {
            "model": "soft", "gamma": model["gamma"],
            "train_error": float(np.mean(train_labels != y_train)),
            "test_error": float(np.mean(test_labels != y_test)),
            "support_count": int(np.sum(model["support"])),
            "free_count": int(np.sum(model["free"])),
            "beta0": model["beta0"], "slack_sum": float(np.sum(model["slacks"])),
            "v_feas": model["v_feas"], "v_gap": model["v_gap"],
        }
        rows.append(row)
        solver = model["solver"]
        checks_pass = model["v_feas"] <= CHECK_TOL and model["v_gap"] <= CHECK_TOL
        failed = failed or not checks_pass or not solver.success
        log.extend([
            f"gamma={model['gamma']:g}",
            f"train_error={row['train_error']:.12g}; test_error={row['test_error']:.12g}; "
            f"support_count={row['support_count']}; free_count={row['free_count']}",
            f"beta0={row['beta0']:.12g}; slack_sum={row['slack_sum']:.12g}",
            f"primal={model['primal']:.17g}; dual={model['dual']:.17g}",
            f"v_feas={model['v_feas']:.12g}; v_gap={model['v_gap']:.12g}; checks_pass={checks_pass}",
            f"success={solver.success}; status={solver.status}; iterations={solver.nit}; "
            f"message={solver.message}",
            "",
        ])

    # Include the required reference row for the final CSV format.
    rows.append(hard_reference_row(folder / "data/Q6_hard_summary.csv"))
    with (folder / "Q6_results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                key: format(value, ".17g") if isinstance(value, float) else value
                for key, value in row.items()
            })
    log.append("Hard reference row copied from Q6_hard_summary.csv; no hard-margin fit performed.")
    with (folder / "data/Q6_train.csv").open(newline="", encoding="utf-8-sig") as handle:
        train_ids = next(csv.reader(handle))[1:]
    X_grid, hard_scores = load_grid(folder / "data/Q6_hard_grid.csv")
    plot_boundaries(X_train, y_train, train_ids, models, X_grid, hard_scores,
                    folder / "Q6_boundaries.png")
    log.append("Q6(d): all panels use the supplied 80 by 80 grid; h=0.7.")
    log.append("Circled IDs: " + ", ".join(ALTERED_IDS) + ". Labels are unchanged.")
    log.append("Saved Q6_results.csv, Q6_boundaries.png and Q6_solver_log.txt next to Q6.py.")
    report = "\n".join(log) + "\n"
    (folder / "Q6_solver_log.txt").write_text(report, encoding="utf-8")
    print(report)
    if failed:
        raise RuntimeError("A solver or numerical check failed. See Q6_solver_log.txt; do not alter the problem.")


if __name__ == "__main__":
    main()


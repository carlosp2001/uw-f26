"""STAT 841 / CM 763, Assignment 2: Question 1(a)-(b)."""

from pathlib import Path
import csv

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

N = 80
H = 0.12
SIGMA = 0.3
SEED = 841
MODEL_SIZES = (5, 9, 17, 33)
OUT = Path(__file__).resolve().parent
COLUMNS = ("m", "singular_ratio", "empirical_error", "sure", "latent_error")


def true_function(x: np.ndarray) -> np.ndarray:
    """Evaluate the true function prescribed in Question 1."""
    return np.sin(2 * np.pi * x) + 0.5 * np.cos(5 * np.pi * x)


def design_matrix(x: np.ndarray, m: int, h: float = H) -> np.ndarray:
    """Return Phi with shape (m, n): one row per RBF, one column per input."""
    x = np.asarray(x, dtype=float)
    if x.ndim != 1 or x.size == 0 or not np.all(np.isfinite(x)):
        raise ValueError("x must be a nonempty, finite one-dimensional array.")
    if m < 2 or not np.isfinite(h) or h <= 0:
        raise ValueError("m must be at least 2 and h must be positive and finite.")

    # Equivalent to mu_j = -1 + 2*(j-1)/(m-1), j = 1, ..., m.
    centers = np.linspace(-1.0, 1.0, m)
    differences = x[None, :] - centers[:, None]
    return np.exp(-(differences**2) / h**2)


def least_squares_fit(
        phi: np.ndarray, y: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Fit weights by SVD-based least squares, without intercept or penalty."""
    phi = np.asarray(phi, dtype=float)
    y = np.asarray(y, dtype=float)
    if phi.ndim != 2 or y.ndim != 1 or phi.shape[1] != y.size:
        raise ValueError("Expected Phi with shape (m, n) and y with shape (n,).")
    if not np.all(np.isfinite(phi)) or not np.all(np.isfinite(y)):
        raise ValueError("Phi and y must contain finite values.")

    # Do not form or invert Phi @ Phi.T.
    weights, _, rank, _ = np.linalg.lstsq(phi.T, y, rcond=None)
    if rank != phi.shape[0]:
        raise RuntimeError("The prescribed design should have full row rank.")
    fitted = phi.T @ weights
    return weights, fitted


def error_measures(
        y: np.ndarray, fitted: np.ndarray, f: np.ndarray, m: int
) -> tuple[float, float, float]:
    """Compute e_m, s_m, q_m at the training inputs; retain negative SURE."""
    n = y.size
    empirical_error = float(np.sum((y - fitted) ** 2) / n)
    sure = empirical_error - SIGMA**2 + 2 * SIGMA**2 * m / n
    latent_error = float(np.sum((fitted - f) ** 2) / n)
    return empirical_error, sure, latent_error


def select_model(rows: list[dict], criterion: str) -> int:
    """Minimize the requested criterion, breaking exact ties with smaller m."""
    best = min(rows, key=lambda row: (row[criterion], row["m"]))
    return int(best["m"])


def save_results(rows: list[dict]) -> None:
    """Write the prescribed columns, without an extra index column."""
    with (OUT / "Q1_first.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                name: str(row[name]) if name == "m" else f"{row[name]:.17g}"
                for name in COLUMNS
            })


def save_figure(
        x: np.ndarray,
        y: np.ndarray,
        grid: np.ndarray,
        curves: dict[int, np.ndarray],
        selected: int,
) -> None:
    """Save one figure with four aligned panels as a 300-DPI PNG."""
    fig, axes = plt.subplots(
        2, 2, figsize=(11, 7), sharex=True, sharey=True, constrained_layout=True
    )
    f_grid = true_function(grid)
    for ax, m in zip(axes.flat, MODEL_SIZES):
        ax.scatter(x, y, s=14, alpha=0.6, label="Observed responses")
        ax.plot(grid, f_grid, "--", linewidth=1.7, label="True function")
        ax.plot(grid, curves[m], linewidth=1.8, label=f"RBF fit (m={m})")
        suffix = " -- SURE selected" if m == selected else ""
        ax.set_title(f"m = {m}{suffix}")
        ax.set(xlim=(-1, 1), xlabel="x", ylabel="Response")
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8, loc="lower left")
    fig.savefig(OUT / "Q1_first_figure.png", dpi=300)
    plt.close(fig)


def main() -> None:
    # Equivalent to x_i = -1 + 2*(i-1)/(N-1), i = 1, ..., N.
    x = np.linspace(-1.0, 1.0, N)
    f = true_function(x)

    # Generate ONE response vector shared by all four models.
    # For part (c), retain this vector as repetition 1 and continue this RNG.
    rng = np.random.default_rng(SEED)
    noise = rng.normal(loc=0.0, scale=SIGMA, size=N)
    y = f + noise

    grid = np.linspace(-1.0, 1.0, 1000)
    rows = []
    curves = {}

    print(f"Seed: {SEED}; n: {N}; h: {H}; noise standard deviation: {SIGMA}")
    print("Q1(a): NumPy shapes (mathematical vectors are column vectors)")
    for m in MODEL_SIZES:
        phi = design_matrix(x, m)
        weights, fitted = least_squares_fit(phi, y)
        singular_values = np.linalg.svd(phi, compute_uv=False)
        ratio = float(singular_values[-1] / singular_values[0])
        e, s, q = error_measures(y, fitted, f, m)
        rows.append(dict(
            m=m, singular_ratio=ratio, empirical_error=e, sure=s, latent_error=q
        ))

        # Use exactly the same centers, width, and learned weights on the grid.
        curves[m] = design_matrix(grid, m).T @ weights
        print(
            f"m={m:2d}: Phi.T={phi.T.shape}, w={weights.shape}, "
            f"y={y.shape}, fitted={fitted.shape}, ratio={ratio:.12g}"
        )

    selected = select_model(rows, "sure")
    save_results(rows)
    save_figure(x, y, grid, curves, selected)

    print("\nQ1(b): errors at the 80 training inputs")
    print("m   singular_ratio   empirical_error   sure            latent_error")
    for row in rows:
        print(f"{row['m']:2d}  " + "  ".join(
            f"{row[name]:.10f}" for name in COLUMNS[1:]
        ))
    print(f"SURE-selected m: {selected}")
    print(f"Empirical-error minimum m (comparison only): {select_model(rows, 'empirical_error')}")
    print(f"True-function-error minimum m (comparison only): {select_model(rows, 'latent_error')}")
    print("Saved Q1_first.csv and Q1_first_figure.png next to Q1.py.")


if __name__ == "__main__":
    main()

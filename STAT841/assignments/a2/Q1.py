"""STAT 841 / CM 763, Assignment 2: Question 1(a)-(c)."""

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
REPETITIONS = 200
MODEL_SIZES = (5, 9, 17, 33)
OUT = Path(__file__).resolve().parent
COLUMNS = ("m", "singular_ratio", "empirical_error", "sure", "latent_error")
SIMULATION_COLUMNS = (
    "m", "mean_empirical_error", "mean_sure", "mean_latent_error",
    "mean_difference", "mcse_difference", "selection_count",
)


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


def repeated_experiments(
        x: np.ndarray,
        f: np.ndarray,
        rng: np.random.Generator,
        first_rows: list[dict],
) -> list[dict]:
    """Reuse part (b) as repetition 1, then draw 199 new noise vectors."""
    b = REPETITIONS
    number_of_models = len(MODEL_SIZES)

    # Each row is a repetition; each column is a model size.
    empirical = np.empty((b, number_of_models))
    sure = np.empty((b, number_of_models))
    latent = np.empty((b, number_of_models))

    # Retain the results already calculated from the first response vector.
    for j, row in enumerate(first_rows):
        empirical[0, j] = row["empirical_error"]
        sure[0, j] = row["sure"]
        latent[0, j] = row["latent_error"]

    # Inputs, centers, and width stay fixed across all repetitions.
    designs = [design_matrix(x, m) for m in MODEL_SIZES]

    for t in range(1, b):
        # Continue the existing generator: do not reset the seed here.
        y = f + rng.normal(loc=0.0, scale=SIGMA, size=x.size)

        # All four models use this same response vector.
        for j, m in enumerate(MODEL_SIZES):
            _, fitted = least_squares_fit(designs[j], y)
            empirical[t, j], sure[t, j], latent[t, j] = error_measures(
                y, fitted, f, m
            )

    selection_counts = np.zeros(number_of_models, dtype=int)
    for t in range(b):
        # argmin returns the first minimum. MODEL_SIZES is increasing,
        # so exact ties select the smaller m. Negative SURE is retained.
        selected_index = int(np.argmin(sure[t]))
        selection_counts[selected_index] += 1

    rows = []
    for j, m in enumerate(MODEL_SIZES):
        differences = sure[:, j] - latent[:, j]
        mean_difference = float(np.mean(differences))

        # Use the MCSE formula in the assignment, including b - 1.
        sample_variance = np.sum((differences - mean_difference) ** 2) / (b - 1)
        mcse = float(np.sqrt(sample_variance) / np.sqrt(b))

        rows.append(dict(
            m=m,
            mean_empirical_error=float(np.mean(empirical[:, j])),
            mean_sure=float(np.mean(sure[:, j])),
            mean_latent_error=float(np.mean(latent[:, j])),
            mean_difference=mean_difference,
            mcse_difference=mcse,
            selection_count=int(selection_counts[j]),
        ))
    return rows


def save_simulation_results(rows: list[dict]) -> None:
    """Save the four simulation rows using the required CSV layout."""
    with (OUT / "Q1_simulation.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SIMULATION_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                name: str(row[name]) if name in ("m", "selection_count")
                else f"{row[name]:.17g}"
                for name in SIMULATION_COLUMNS
            })


def save_simulation_figure(rows: list[dict]) -> None:
    """Plot the three average errors against m in one 300-DPI PNG."""
    sizes = [row["m"] for row in rows]
    fig, ax = plt.subplots(figsize=(8, 4.8), constrained_layout=True)
    ax.plot(sizes, [row["mean_empirical_error"] for row in rows],
            marker="o", label="Average training error")
    ax.plot(sizes, [row["mean_sure"] for row in rows],
            marker="s", linestyle="--", label="Average SURE")
    ax.plot(sizes, [row["mean_latent_error"] for row in rows],
            marker="^", linestyle=":", label="Average true-function error")
    ax.set_xticks(sizes)
    ax.set_xlabel("Number of RBF units (m)")
    ax.set_ylabel("Average error")
    ax.set_title(f"RBF regression: {REPETITIONS} repetitions (seed {SEED})")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.savefig(OUT / "Q1_simulation_figure.png", dpi=300)
    plt.close(fig)


def main() -> None:
    # Equivalent to x_i = -1 + 2*(i-1)/(N-1), i = 1, ..., N.
    x = np.linspace(-1.0, 1.0, N)
    f = true_function(x)

    # Generate ONE response vector shared by all four models.
    # Part (c) retains this vector as repetition 1 and continues this RNG.
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

    # Question 1(c): the generator has already drawn the noise for part (b).
    simulation_rows = repeated_experiments(x, f, rng, rows)
    save_simulation_results(simulation_rows)
    save_simulation_figure(simulation_rows)

    print("\nQ1(c): averages over 200 repetitions, including part (b)")
    print("m   mean_e       mean_s       mean_q       mean_d       mcse         count")
    for row in simulation_rows:
        values = "  ".join(f"{row[name]:.8f}" for name in SIMULATION_COLUMNS[1:-1])
        print(f"{row['m']:2d}  {values}  {row['selection_count']:3d}")
    print("Selection count total:", sum(row["selection_count"] for row in simulation_rows))
    print("Saved Q1_simulation.csv and Q1_simulation_figure.png next to Q1.py.")


if __name__ == "__main__":
    main()

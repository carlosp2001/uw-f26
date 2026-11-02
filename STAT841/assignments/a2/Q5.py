"""STAT 841 / CM 763 - Assignment 2, Q5(a)-(c).

Implement the supplied kernels and prediction rules; check Gram symmetry.
Save the error table and the four-panel boundary figure, without fitting any model.
Observations are columns. Input and output paths are relative to this file.
"""

import csv
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def read_numeric_rows(path: Path) -> tuple[list[str], dict[str, np.ndarray]]:
    """Read IDs and named numeric rows using the assignment's CSV layout."""
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.reader(handle))
    if len(rows) < 2:
        raise ValueError(f"Empty or incomplete input file: {path.name}")

    ids = rows[0][1:]
    if not ids or len(set(ids)) != len(ids):
        raise ValueError(f"Missing or duplicate observation IDs: {path.name}")

    values = {}
    for row in rows[1:]:
        if len(row) != len(ids) + 1 or row[0] in values:
            raise ValueError(f"Invalid or duplicate numeric row: {path.name}")
        values[row[0]] = np.asarray(row[1:], dtype=float)
        if not np.all(np.isfinite(values[row[0]])):
            raise ValueError(f"Non-finite values: {path.name}")
    return ids, values


def read_observations(path: Path) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Return observation IDs, X with shape (2, n), and y with shape (n,)."""
    ids, values = read_numeric_rows(path)
    if not {"x1", "x2", "y"}.issubset(values):
        raise ValueError(f"Required rows x1, x2, and y not found: {path.name}")
    X = np.vstack([values["x1"], values["x2"]])
    y = values["y"]
    if not np.all(np.isin(y, [-1, 1])):
        raise ValueError(f"Labels must be -1 or +1: {path.name}")
    return ids, X, y


def kernel_matrix(
    X: np.ndarray,
    Z: np.ndarray,
    kernel: str,
    h: float | None = None,
) -> np.ndarray:
    """Return K[i, j] = k(X[:, i], Z[:, j]), with shape (n, p)."""
    X = np.asarray(X, dtype=float)
    Z = np.asarray(Z, dtype=float)
    if X.ndim != 2 or Z.ndim != 2 or X.shape[0] != Z.shape[0]:
        raise ValueError("X and Z must have shapes (d, n) and (d, p).")
    if not np.all(np.isfinite(X)) or not np.all(np.isfinite(Z)):
        raise ValueError("X and Z must contain finite values.")

    if kernel == "quadratic":
        return (X.T @ Z) ** 2

    if kernel != "gaussian":
        raise ValueError("kernel must be 'quadratic' or 'gaussian'.")
    if h is None or not np.isfinite(h) or h <= 0:
        raise ValueError("A Gaussian kernel requires a positive width h.")

    K = np.empty((X.shape[1], Z.shape[1]))
    for j in range(Z.shape[1]):
        # Compare one new observation with all training observations.
        differences = X - Z[:, j:j + 1]
        squared_distances = np.sum(differences ** 2, axis=0)
        K[:, j] = np.exp(-squared_distances / h ** 2)
    return K


def align_coefficients(
    train_ids: list[str],
    coefficient_ids: list[str],
    alpha: np.ndarray,
) -> np.ndarray:
    """Reorder one supplied multiplier row to match training column IDs."""
    alpha = np.asarray(alpha, dtype=float)
    if alpha.shape != (len(coefficient_ids),):
        raise ValueError("There must be one multiplier per coefficient ID.")
    if len(set(train_ids)) != len(train_ids):
        raise ValueError("Training observation IDs must be unique.")
    if len(set(coefficient_ids)) != len(coefficient_ids):
        raise ValueError("Coefficient column IDs must be unique.")
    if set(train_ids) != set(coefficient_ids):
        raise ValueError("Training and coefficient IDs must match exactly.")

    positions = {obs_id: j for j, obs_id in enumerate(coefficient_ids)}
    return alpha[[positions[obs_id] for obs_id in train_ids]]


def predict(
    X: np.ndarray,
    y: np.ndarray,
    alpha: np.ndarray,
    beta0: float,
    Z: np.ndarray,
    kernel: str,
    h: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return scores and labels for the columns of Z, without fitting.

    X, y, and alpha must share the same training-observation order.
    Use align_coefficients before this function if necessary.
    """
    C = kernel_matrix(X, Z, kernel, h)
    y = np.asarray(y, dtype=float)
    alpha = np.asarray(alpha, dtype=float)
    if y.shape != (C.shape[0],) or alpha.shape != (C.shape[0],):
        raise ValueError("y and alpha must have one entry per training column.")
    if not np.all(np.isin(y, [-1, 1])):
        raise ValueError("Training labels must be -1 or +1.")
    if not np.all(np.isfinite(alpha)) or not np.isfinite(beta0):
        raise ValueError("Multipliers and intercept must be finite.")

    # Multiply the supplied alpha values by y exactly once.
    scores = C.T @ (alpha * y) + beta0
    predictions = np.where(scores >= 0, 1, -1)
    return scores, predictions


def plot_boundaries(
    X: np.ndarray,
    y: np.ndarray,
    models: list[dict],
    output_path: Path,
) -> None:
    """Plot all four supplied classifiers in the original input coordinates."""
    # The assignment requires at least 80 points per axis.
    axis = np.linspace(-2.2, 2.2, 300)
    xx, yy = np.meshgrid(axis, axis)
    grid = np.vstack([xx.ravel(), yy.ravel()])

    fig, axes = plt.subplots(2, 2, figsize=(11, 10), sharex=True, sharey=True)
    fig.subplots_adjust(
        left=0.08, right=0.98, bottom=0.14, top=0.95, wspace=0.17, hspace=0.23
    )

    for ax, model in zip(axes.flat, models):
        scores, labels = predict(
            X, y, model["alpha"], model["beta0"], grid,
            model["kernel"], model["h"]
        )
        score_grid = scores.reshape(xx.shape)
        label_grid = labels.reshape(xx.shape)

        # Use the same colors for each class in every panel.
        ax.contourf(
            xx, yy, label_grid, levels=[-1.5, 0, 1.5],
            colors=["C0", "C1"], alpha=0.16
        )

        # Draw and label the prescribed score contours where they occur.
        for level, style, width in [(-1, "--", 1.1), (0, "-", 1.8), (1, "-.", 1.1)]:
            if score_grid.min() < level < score_grid.max():
                contour = ax.contour(
                    xx, yy, score_grid, levels=[level], colors="black",
                    linestyles=style, linewidths=width
                )
                contour_label = {level: f"g = {level:+d}" if level != 0 else "g = 0"}
                labels = ax.clabel(contour, fmt=contour_label, fontsize=9, inline=True)
                if not labels:
                    # Tiny closed contours may need an external label.
                    segment = max(contour.allsegs[0], key=len)
                    point = segment[np.argmax(segment[:, 0])]
                    offset = (20, 25) if level == -1 else (-12, -32)
                    ax.annotate(
                        contour_label[level], xy=point, xytext=offset,
                        textcoords="offset points", fontsize=9,
                        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85},
                        arrowprops={"arrowstyle": "->", "linewidth": 0.8}
                    )

        for label, marker, color in [(-1, "o", "C0"), (1, "^", "C1")]:
            mask = y == label
            ax.scatter(
                X[0, mask], X[1, mask], c=color, marker=marker,
                s=35, edgecolors="white", linewidths=0.4, zorder=3
            )

        support = model["alpha"] > 1e-6
        ax.scatter(
            X[0, support], X[1, support], s=100,
            facecolors="none", edgecolors="black", linewidths=1.3, zorder=4
        )

        title = "Quadratic" if model["kernel"] == "quadratic" else f"Gaussian (h = {model['h']:g})"
        ax.set_title(title, fontsize=14)
        ax.set_xlim(-2.2, 2.2)
        ax.set_ylim(-2.2, 2.2)
        ax.set_aspect("equal")
        ax.set_xlabel(r"$x_1$", fontsize=13)
        ax.set_ylabel(r"$x_2$", fontsize=13)
        ax.set_xticks([-2, -1, 0, 1, 2])
        ax.set_yticks([-2, -1, 0, 1, 2])
        ax.tick_params(labelbottom=True, labelleft=True)

    handles = [
        Line2D([], [], marker="o", color="C0", linestyle="none", label="Class -1", markersize=7),
        Line2D([], [], marker="^", color="C1", linestyle="none", label="Class +1", markersize=7),
        Line2D([], [], marker="o", color="black", markerfacecolor="none", linestyle="none", label="Support vector", markersize=10),
        Line2D([], [], color="black", linestyle="--", label="g = -1"),
        Line2D([], [], color="black", linestyle="-", label="g = 0"),
        Line2D([], [], color="black", linestyle="-.", label="g = +1"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=11, bbox_to_anchor=(0.5, 0.015))
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """Evaluate all four supplied models without changing their parameters."""
    folder = Path(__file__).resolve().parent
    train_ids, X, y = read_observations(folder / "data/Q5_train.csv")
    _, Z, y_test = read_observations(folder / "data/Q5_test.csv")
    coefficient_ids, coefficient_rows = read_numeric_rows(folder / "data/Q5_models.csv")

    with (folder / "data/Q5_model_info.csv").open(newline="", encoding="utf-8-sig") as handle:
        model_info = {row["model"]: row for row in csv.DictReader(handle)}

    # Use the prescribed results-file order, not test performance.
    model_names = ["quadratic", "gaussian_02", "gaussian_07", "gaussian_14"]
    results = []
    models = []

    print(f"Training X: {X.shape}; test Z: {Z.shape}")
    print("Q5(a): training Gram and cross-kernel matrices")
    for name in model_names:
        info = model_info[name]
        kernel = info["kernel"]
        h = float(info["h"]) if info["h"].strip() else None
        beta0 = float(info["beta0"])

        # Match alpha to the training IDs; do not change the supplied values.
        alpha = align_coefficients(
            train_ids, coefficient_ids, coefficient_rows["alpha_" + name]
        )

        models.append({
            "name": name, "kernel": kernel, "h": h,
            "beta0": beta0, "alpha": alpha
        })

        K = kernel_matrix(X, X, kernel, h)
        C = kernel_matrix(X, Z, kernel, h)
        symmetry_error = float(np.max(np.abs(K - K.T)))

        # Predict labels using all supplied coefficients, not only support vectors.
        _, train_predictions = predict(X, y, alpha, beta0, X, kernel, h)
        _, test_predictions = predict(X, y, alpha, beta0, Z, kernel, h)

        # These quantities are also the required table for part (b).
        train_error = float(np.mean(train_predictions != y))
        test_error = float(np.mean(test_predictions != y_test))
        support_count = int(np.count_nonzero(alpha > 1e-6))
        results.append((name, train_error, test_error, support_count, symmetry_error))

        print(
            f"{name:12s}: K={K.shape}, C={C.shape}, "
            f"symmetry error={symmetry_error:.12g}"
        )

    # Write the exact column names and order required for Q5_results.csv.
    with (folder / "Q5_results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "model", "train_error", "test_error", "support_count", "symmetry_error"
        ])
        for name, train_error, test_error, count, symmetry_error in results:
            writer.writerow([
                name, f"{train_error:.17g}", f"{test_error:.17g}",
                count, f"{symmetry_error:.17g}"
            ])

    print("\nQ5(b): errors, support counts, and symmetry checks")
    print("model          train_error   test_error   support_count")
    for name, train_error, test_error, count, _ in results:
        print(f"{name:12s}  {train_error:.8f}    {test_error:.8f}    {count}")
    plot_boundaries(X, y, models, folder / "Q5_boundaries.png")
    print("\nQ5(c): saved Q5_boundaries.png (300 x 300 grid, 300 DPI).")
    print("Saved Q5_results.csv next to Q5.py. No model fitting was performed.")


if __name__ == "__main__":
    main()

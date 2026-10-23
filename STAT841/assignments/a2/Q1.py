"""STAT 841 / CM 763 - Assignment 2, Question 1(a)."""
import numpy as np

N = 80
H = 0.12
SIGMA = 0.3
SEED = 841
MODEL_SIZES = (5, 9, 17, 33)


def true_function(x):
    """Evaluate the regression function at a vector of inputs."""
    return np.sin(2 * np.pi * x) + 0.5 * np.cos(5 * np.pi * x)


def design_matrix(x, m, h=H):
    """Return the RBF design matrix Phi with shape (m, n)."""
    centers = np.linspace(-1.0, 1.0, m)
    phi = np.exp(-((x[None, :] - centers[:, None]) ** 2) / h ** 2)
    return phi


def least_squares_fit(phi, y):
    """Fit the RBF weights using an SVD-based least-squares solve."""
    weights, _, _, _ = np.linalg.lstsq(phi.T, y, rcond=None)
    fitted = phi.T @ weights
    return weights, fitted


def main():
    x = np.linspace(-1.0, 1.0, N)
    rng = np.random.default_rng(SEED)
    y = true_function(x) + rng.normal(0.0, SIGMA, size=N)
    print(f"Random seed: {SEED}")
    print(
        "m  Phi.shape  Phi.T.shape  w.shape  y.shape  f_hat.shape  singular_ratio")
    for m in MODEL_SIZES:
        phi = design_matrix(x, m)
        weights, fitted = least_squares_fit(phi, y)
        singular_values = np.linalg.svd(phi, compute_uv=False)
        singular_ratio = singular_values[-1] / singular_values[0]
        print(f"{m:2d} {str(phi.shape):>10} {str(phi.T.shape):>12} "
              f"{str(weights.shape):>9} {str(y.shape):>9} "
              f"{str(fitted.shape):>12} {singular_ratio:.12g}")


if __name__ == "__main__":
    main()

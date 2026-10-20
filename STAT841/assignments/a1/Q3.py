"""STAT 841 Assignment 1 - Question 3(e): PCA versus Fisher.

Dependencies: numpy, matplotlib
Run: python Q3.py
Observations are columns (d x n). No prebuilt PCA/Fisher/CV routines.
"""
from pathlib import Path
import numpy as np

# Uncomment to visualize the dataset generated
# import matplotlib

# matplotlib.use('MacOSX')

import matplotlib.pyplot as plt

SEED = 42
N_PER_CLASS = 100
D = 10
A = 2.0
V = 300.0
N_SPLITS = 20
TRAIN_PER_CLASS = 70


def generate_dataset(rng):
    mean_plus = np.zeros(D)
    mean_plus[1] = A
    covariance = np.diag([V] + [1.0] * (D - 1))
    minus = rng.multivariate_normal(-mean_plus, covariance, size=N_PER_CLASS)
    plus = rng.multivariate_normal(mean_plus, covariance, size=N_PER_CLASS)
    X = np.vstack((minus, plus)).T
    y = np.concatenate((np.full(N_PER_CLASS, -1), np.full(N_PER_CLASS, 1)))
    return X, y


def stratified_split(y, rng):
    minus = rng.permutation(np.flatnonzero(y == -1))
    plus = rng.permutation(np.flatnonzero(y == 1))
    train = np.concatenate((minus[:TRAIN_PER_CLASS], plus[:TRAIN_PER_CLASS]))
    test = np.concatenate((minus[TRAIN_PER_CLASS:], plus[TRAIN_PER_CLASS:]))
    return train, test


def center_data(X_train, X_test):
    mean = X_train.mean(axis=1, keepdims=True)
    return X_train - mean, X_test - mean


def compute_pca_direction(X_train_centered):
    covariance = X_train_centered @ X_train_centered.T / X_train_centered.shape[1]
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    direction = eigenvectors[:, np.argmax(eigenvalues)]
    return direction / np.linalg.norm(direction)


def compute_fisher_direction(X_train_centered, y_train):
    X_minus = X_train_centered[:, y_train == -1]
    X_plus = X_train_centered[:, y_train == 1]
    mean_minus = X_minus.mean(axis=1)
    mean_plus = X_plus.mean(axis=1)
    C_minus = (X_minus - mean_minus[:, None]) @ (X_minus - mean_minus[:, None]).T / X_minus.shape[1]
    C_plus = (X_plus - mean_plus[:, None]) @ (X_plus - mean_plus[:, None]).T / X_plus.shape[1]
    pooled_sum = C_plus + C_minus
    # A linear solve avoids explicitly forming a matrix inverse.
    direction = np.linalg.solve(pooled_sum, mean_plus - mean_minus)
    norm = np.linalg.norm(direction)
    if not np.isfinite(norm) or norm <= 0:
        raise ValueError('Invalid Fisher direction')
    return direction / norm


def predict_nearest_class_mean(train_scores, y_train, test_scores):
    mean_plus = train_scores[y_train == 1].mean()
    mean_minus = train_scores[y_train == -1].mean()
    midpoint = (mean_plus + mean_minus) / 2.0
    return np.where((mean_plus - mean_minus) * (test_scores - midpoint) > 0, 1, -1)


def classification_error(y_true, y_pred):
    return float(np.mean(y_true != y_pred))


def run_experiment(X, y, rng):
    errors_pca = []
    errors_fisher = []

    for _ in range(N_SPLITS):
        train_idx, test_idx = stratified_split(y, rng)
        X_train, X_test = center_data(X[:, train_idx], X[:, test_idx])
        y_train, y_test = y[train_idx], y[test_idx]

        for direction_function, results in ((compute_pca_direction, errors_pca),
                                            (lambda x: compute_fisher_direction(x, y_train), errors_fisher)):
            w = direction_function(X_train)
            train_scores = w @ X_train
            test_scores = w @ X_test
            prediction = predict_nearest_class_mean(train_scores, y_train, test_scores)
            results.append(classification_error(y_test, prediction))

    return np.asarray(errors_pca), np.asarray(errors_fisher)


def save_comparison_figure(errors_pca, errors_fisher, path):
    split_numbers = np.arange(1, N_SPLITS + 1)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(split_numbers, errors_pca, marker='o', label='PCA (PC1)')
    ax.plot(split_numbers, errors_fisher, marker='s', label='Fisher')
    ax.set(xlabel='Stratified split', ylabel='Test classification error',
           title='Question 3(e): Test error over 20 stratified splits')
    ax.set_xticks(split_numbers)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_dataset(X: np.ndarray, y: np.ndarray) -> None:
    """Visualize the first two features of the generated dataset."""

    # Select observations belonging to each class
    negative_mask = y == -1
    positive_mask = y == 1

    plt.figure(figsize=(11, 6))

    # Class -1
    plt.scatter(
        X[0, negative_mask],
        X[1, negative_mask],
        color="royalblue",
        alpha=0.65,
        label="Class -1",
        s=35
    )

    # Class +1
    plt.scatter(
        X[0, positive_mask],
        X[1, positive_mask],
        color="tomato",
        alpha=0.65,
        label="Class +1",
        s=35
    )

    # Population class means
    plt.scatter(
        [0, 0],
        [-A, A],
        color="black",
        marker="x",
        s=130,
        linewidths=2,
        label="Population class means"
    )

    plt.axhline(0, color="gray", linestyle="--", alpha=0.5)
    plt.axvline(0, color="gray", linestyle="--", alpha=0.5)

    plt.xlabel("Feature X1 (variance = 300)")
    plt.ylabel("Feature X2 (class separation)")

    plt.title("Question 3(e): Generated Gaussian Dataset")

    plt.legend()
    plt.grid(alpha=0.25)

    plt.tight_layout()
    plt.savefig("Q3_dataset.png", dpi=300)
    plt.show()


def main():
    rng = np.random.default_rng(SEED)
    X, y = generate_dataset(rng)

    # Uncomment to visualize the original dataset
    # plot_dataset(X, y)
    errors_pca, errors_fisher = run_experiment(X, y, rng)
    output = Path(__file__).resolve().parent / 'Q3_comparison.png'
    save_comparison_figure(errors_pca, errors_fisher, output)
    print(f'Random seed: {SEED}')
    print('Dataset: n=200, d=10, a=2, v=300; 100 observations per class')
    print('20 stratified splits: 70 train and 30 test per class')
    print('Method          Mean test error    Sample SD (ddof=1)')
    print(f'PCA (PC1)       {errors_pca.mean():.6f}           {errors_pca.std(ddof=1):.6f}')
    print(f'Fisher          {errors_fisher.mean():.6f}           {errors_fisher.std(ddof=1):.6f}')
    print(f'Figure: {output.name}')


if __name__ == '__main__':
    main()

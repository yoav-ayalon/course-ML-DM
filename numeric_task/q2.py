
# q2_simulation.py
# Combined from Jupyter cells into a single Python script.
# Requires: numpy, pandas, scikit-learn (for the optional Q2d section), and CSV files: M1.csv, M2.csv, Sigma1.csv, Sigma2.csv

import numpy as np
import pandas as pd
from pathlib import Path

# Optional (used in Q2d placeholder)
try:
    from sklearn.ensemble import RandomForestClassifier  # noqa: F401
    from sklearn.metrics import accuracy_score           # noqa: F401
except Exception:
    pass

# -----------------------------
# Global RNG for reproducibility
# -----------------------------
RNG = np.random.default_rng(42)


def mle_cov(X: np.ndarray) -> np.ndarray:
    """Maximum-likelihood covariance (divide by n)."""
    n = X.shape[0]
    Xc = X - X.mean(axis=0, keepdims=True)
    return (Xc.T @ Xc) / n


def main():
    # ======================
    # Load parameters (resolve CSV paths relative to this script so q2.py
    # can be executed from any working directory)
    # ======================
    base_dir = Path(__file__).resolve().parent
    try:
        mu1 = pd.read_csv(base_dir / "M1.csv", header=None).values.squeeze()
        mu2 = pd.read_csv(base_dir / "M2.csv", header=None).values.squeeze()
        Sigma1 = pd.read_csv(base_dir / "Sigma1.csv", header=None).values
        Sigma2 = pd.read_csv(base_dir / "Sigma2.csv", header=None).values
    except FileNotFoundError as e:
        missing = e.filename if getattr(e, 'filename', None) else str(e)
        raise FileNotFoundError(
            f"Could not find required CSV file {missing}.\n"
            f"Expected the files to be next to {Path(__file__).name} in {base_dir}.\n"
            "Place M1.csv, M2.csv, Sigma1.csv and Sigma2.csv in that directory or run the script from there."
        ) from e

    print("mu1 shape:", mu1.shape)
    print("Sigma1 shape:", Sigma1.shape)

    # ======================
    # Q2a — Generate dataset
    # ======================
    P1, P2 = 0.35, 0.65
    N_total = 10_000

    n1 = int(round(P1 * N_total))
    n2 = N_total - n1
    print(f"Class 1: {n1} samples | Class 2: {n2} samples")

    X1 = RNG.multivariate_normal(mean=mu1, cov=Sigma1, size=n1)
    X2 = RNG.multivariate_normal(mean=mu2, cov=Sigma2, size=n2)

    print(X1.shape, X2.shape)

    y1 = np.zeros(n1, dtype=int)
    y2 = np.ones(n2, dtype=int)

    X = np.vstack([X1, X2])
    y = np.concatenate([y1, y2])

    # shuffle
    perm = RNG.permutation(N_total)
    X, y = X[perm], y[perm]

    df = pd.DataFrame(X, columns=[f"x{i+1}" for i in range(X.shape[1])])
    df["label"] = y

    print(df.head())
    print(df["label"].value_counts(normalize=True).rename("proportion"))

    # ======================
    # Q2b — MLE estimates & errors
    # ======================
    X1_cls = df[df.label == 0].iloc[:, :-1].to_numpy()
    X2_cls = df[df.label == 1].iloc[:, :-1].to_numpy()

    # MLE mean
    mu1_hat = X1_cls.mean(axis=0)
    mu2_hat = X2_cls.mean(axis=0)

    # MLE covariance
    Sigma1_hat = mle_cov(X1_cls)
    Sigma2_hat = mle_cov(X2_cls)

    # Load true parameters (again for clarity)
    mu1_true = mu1
    mu2_true = mu2
    Sigma1_true = Sigma1
    Sigma2_true = Sigma2

    # Compute errors
    mu1_error = np.linalg.norm(mu1_hat - mu1_true)
    mu2_error = np.linalg.norm(mu2_hat - mu2_true)
    Sigma1_error = np.linalg.norm(Sigma1_hat - Sigma1_true, ord='fro')
    Sigma2_error = np.linalg.norm(Sigma2_hat - Sigma2_true, ord='fro')

    np.set_printoptions(precision=4, suppress=True)
    print("\nClass 1:")
    print("mu1_hat =", mu1_hat)
    print("Sigma1_hat =\n", Sigma1_hat)
    print("||mu1_hat - mu1_true||_2 =", mu1_error)
    print("||Sigma1_hat - Sigma1_true||_F =", Sigma1_error)

    print("\nClass 2:")
    print("mu2_hat =", mu2_hat)
    print("Sigma2_hat =\n", Sigma2_hat)
    print("||mu2_hat - mu2_true||_2 =", mu2_error)
    print("||Sigma2_hat - Sigma2_true||_F =", Sigma2_error)

    # ======================
    # Q2c — Validation set
    # ======================
    N_val = 2000
    P1_val, P2_val = 0.35, 0.65
    n1_val = int(round(P1_val * N_val))
    n2_val = N_val - n1_val

    X1_val = RNG.multivariate_normal(mean=mu1_true, cov=Sigma1_true, size=n1_val)
    X2_val = RNG.multivariate_normal(mean=mu2_true, cov=Sigma2_true, size=n2_val)

    y1_val = np.zeros(n1_val, dtype=int)
    y2_val = np.ones(n2_val, dtype=int)

    X_val = np.vstack([X1_val, X2_val])
    y_val = np.concatenate([y1_val, y2_val])

    # shuffle validation
    perm_val = RNG.permutation(N_val)
    X_val, y_val = X_val[perm_val], y_val[perm_val]

    df_val = pd.DataFrame(X_val, columns=[f"x{i+1}" for i in range(X_val.shape[1])])
    df_val["label"] = y_val

    print(f"Validation set generated: total={len(df_val)}, class counts =")
    print(df_val.label.value_counts())

    # ======================
    # Q2d — (Placeholder) RF grid
    # ======================
    # You can continue from here to train/evaluate a RandomForest on (X, y) and validate on (X_val, y_val).
    param_grid = [
        {"n_estimators": 50,  "max_depth": None, "max_features": "sqrt"},
        {"n_estimators": 100, "max_depth": None, "max_features": "sqrt"},
        {"n_estimators": 200, "max_depth": None, "max_features": "sqrt"},
        {"n_estimators": 100, "max_depth": 10,   "max_features": "sqrt"},
        {"n_estimators": 100, "max_depth": 20,   "max_features": "sqrt"},
        {"n_estimators": 100, "max_depth": None, "max_features": 0.5},
        {"n_estimators": 200, "max_depth": 20,   "max_features": 0.5},
    ]
    print("\nQ2d param_grid defined (add training/evaluation as needed).\n")

if __name__ == "__main__":
    main()

import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier 
from sklearn.metrics import accuracy_score           


RNG = np.random.default_rng(42)


def mle_cov(X: np.ndarray) -> np.ndarray:
    """Maximum-likelihood covariance (divide by n)."""
    n = X.shape[0]
    Xc = X - X.mean(axis=0, keepdims=True)
    return (Xc.T @ Xc) / n


def load_true_parameters(base_dir: Path):
    """Load the true means and covariances from CSV files located next to this script.

    Returns (mu1, mu2, Sigma1, Sigma2)
    """
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

    return mu1, mu2, Sigma1, Sigma2


def generate_dataset(mu1, mu2, Sigma1, Sigma2, P1=0.35, P2=0.65, N_total=10_000, rng=RNG):
    """Generate synthetic dataset with two Gaussian classes.

    Returns a DataFrame `df` with feature columns x1..xd and a `label` column.
    Also returns (X, y) numpy arrays in case they're needed.
    """
    n1 = int(round(P1 * N_total))
    n2 = N_total - n1
    print(f"Class 1: {n1} samples | Class 2: {n2} samples")

    X1 = rng.multivariate_normal(mean=mu1, cov=Sigma1, size=n1)
    X2 = rng.multivariate_normal(mean=mu2, cov=Sigma2, size=n2)

    y1 = np.zeros(n1, dtype=int)
    y2 = np.ones(n2, dtype=int)

    X = np.vstack([X1, X2])
    y = np.concatenate([y1, y2])

    # shuffle
    perm = rng.permutation(N_total)
    X, y = X[perm], y[perm]

    df = pd.DataFrame(X, columns=[f"x{i+1}" for i in range(X.shape[1])])
    df["label"] = y

    print(df.head())
    print(df["label"].value_counts(normalize=True).rename("proportion"))

    return df, X, y


def compute_mle_estimates(df):
    """Given a DataFrame with last column `label`, compute MLE means and covariances for each class.

    Returns a dict with estimates and errors compared to provided true values if present.
    """
    X1_cls = df[df.label == 0].iloc[:, :-1].to_numpy()
    X2_cls = df[df.label == 1].iloc[:, :-1].to_numpy()

    mu1_hat = X1_cls.mean(axis=0)
    mu2_hat = X2_cls.mean(axis=0)

    Sigma1_hat = mle_cov(X1_cls)
    Sigma2_hat = mle_cov(X2_cls)

    return {
        "mu1_hat": mu1_hat,
        "mu2_hat": mu2_hat,
        "Sigma1_hat": Sigma1_hat,
        "Sigma2_hat": Sigma2_hat,
    }


def compute_errors(estimates, true_params):
    """Compute norms between estimates and true parameters.

    true_params is a tuple (mu1_true, mu2_true, Sigma1_true, Sigma2_true)
    """
    mu1_true, mu2_true, Sigma1_true, Sigma2_true = true_params

    mu1_error = np.linalg.norm(estimates["mu1_hat"] - mu1_true)
    mu2_error = np.linalg.norm(estimates["mu2_hat"] - mu2_true)
    Sigma1_error = np.linalg.norm(estimates["Sigma1_hat"] - Sigma1_true, ord='fro')
    Sigma2_error = np.linalg.norm(estimates["Sigma2_hat"] - Sigma2_true, ord='fro')

    return {
        "mu1_error": mu1_error,
        "mu2_error": mu2_error,
        "Sigma1_error": Sigma1_error,
        "Sigma2_error": Sigma2_error,
    }


def print_mle_results(estimates, errors):
    np.set_printoptions(precision=4, suppress=True)
    print("\nClass 1:")
    print("mu1_hat =", estimates["mu1_hat"])
    print("Sigma1_hat =\n", estimates["Sigma1_hat"])
    print("||mu1_hat - mu1_true||_2 =", errors["mu1_error"])
    print("||Sigma1_hat - Sigma1_true||_F =", errors["Sigma1_error"])

    print("\nClass 2:")
    print("mu2_hat =", estimates["mu2_hat"])
    print("Sigma2_hat =\n", estimates["Sigma2_hat"])
    print("||mu2_hat - mu2_true||_2 =", errors["mu2_error"])
    print("||Sigma2_hat - Sigma2_true||_F =", errors["Sigma2_error"])


def generate_validation_set(mu1_true, mu2_true, Sigma1_true, Sigma2_true,
                            N_val=2000, P1_val=0.35, P2_val=0.65, rng=RNG):
    n1_val = int(round(P1_val * N_val))
    n2_val = N_val - n1_val

    X1_val = rng.multivariate_normal(mean=mu1_true, cov=Sigma1_true, size=n1_val)
    X2_val = rng.multivariate_normal(mean=mu2_true, cov=Sigma2_true, size=n2_val)

    y1_val = np.zeros(n1_val, dtype=int)
    y2_val = np.ones(n2_val, dtype=int)

    X_val = np.vstack([X1_val, X2_val])
    y_val = np.concatenate([y1_val, y2_val])

    # shuffle validation
    perm_val = rng.permutation(N_val)
    X_val, y_val = X_val[perm_val], y_val[perm_val]

    df_val = pd.DataFrame(X_val, columns=[f"x{i+1}" for i in range(X_val.shape[1])])
    df_val["label"] = y_val

    print(f"Validation set generated: total={len(df_val)}, class counts =")
    print(df_val.label.value_counts())

    return df_val, X_val, y_val


def define_param_grid():
    return [
        {"n_estimators": 50,  "max_depth": None, "max_features": "sqrt"},
        {"n_estimators": 100, "max_depth": None, "max_features": "sqrt"},
        {"n_estimators": 200, "max_depth": None, "max_features": "sqrt"},
        {"n_estimators": 100, "max_depth": 10,   "max_features": "sqrt"},
        {"n_estimators": 100, "max_depth": 20,   "max_features": "sqrt"},
        {"n_estimators": 100, "max_depth": None, "max_features": 0.5},
        {"n_estimators": 200, "max_depth": 20,   "max_features": 0.5},
    ]


def activate():
    """High-level orchestration function that runs the full pipeline and returns a dict of artifacts.
    This is the function you can call interactively; each step is modular and testable.
    """
    base_dir = Path(__file__).resolve().parent
    mu1, mu2, Sigma1, Sigma2 = load_true_parameters(base_dir)

    print("mu1 shape:", mu1.shape)
    print("Sigma1 shape:", Sigma1.shape)

    df, X, y = generate_dataset(mu1, mu2, Sigma1, Sigma2)

    estimates = compute_mle_estimates(df)
    errors = compute_errors(estimates, (mu1, mu2, Sigma1, Sigma2))
    print_mle_results(estimates, errors)

    df_val, X_val, y_val = generate_validation_set(mu1, mu2, Sigma1, Sigma2)

    param_grid = define_param_grid()
    print("\nQ2d param_grid defined (add training/evaluation as needed).\n")

    return {
        "df": df,
        "X": X,
        "y": y,
        "estimates": estimates,
        "errors": errors,
        "df_val": df_val,
        "X_val": X_val,
        "y_val": y_val,
        "param_grid": param_grid,
    }



if __name__ == "__main__":
    activate()

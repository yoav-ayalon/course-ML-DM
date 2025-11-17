import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from datetime import datetime

from sklearn.ensemble import RandomForestClassifier 
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler           


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
    print(f"Class 1: {n1} samples | Class 2: {n2} samples","\n")

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
    print("\n||mu1_hat - mu1_true||_2 =", errors["mu1_error"])
    print("||Sigma1_hat - Sigma1_true||_F =", errors["Sigma1_error"])

    print("\nClass 2:")
    print("mu2_hat =", estimates["mu2_hat"])
    print("Sigma2_hat =\n", estimates["Sigma2_hat"])
    print("\n||mu2_hat - mu2_true||_2 =", errors["mu2_error"])
    print("||Sigma2_hat - Sigma2_true||_F =", errors["Sigma2_error"])


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


def tune_rf_on_validation(X_train, y_train, X_val, y_val, param_grid):
    """Evaluate Random Forest configurations on validation set and return best configuration.
    Args:
        X_train, y_train: Training set
        X_val, y_val: Validation set
        param_grid: List of configuration dicts to evaluate
        
    Returns:
        tuple: (best_config, best_val_accuracy, results_list)
    """
    best_config = None
    best_val_accuracy = -1
    results = []
    
    print("\nStep 1: Evaluating Random Forest configurations on validation set...")
    print("-" * 70)
    
    for i, config in enumerate(param_grid, 1):
        rf = RandomForestClassifier(random_state=42, **config)
        rf.fit(X_train, y_train)
        y_pred_val = rf.predict(X_val)
        val_accuracy = accuracy_score(y_val, y_pred_val)
        
        results.append({
            "config_id": i,
            "config": config,
            "val_accuracy": val_accuracy
        })
        
        print(f"Config {i}: n_est={config['n_estimators']}, depth={config['max_depth']}, "
              f"features={config['max_features']} => Val Accuracy: {val_accuracy:.4f}")
        
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_config = config
    
    print(f"\nBest configuration on validation set:")
    print(f"  n_estimators={best_config['n_estimators']}, "
          f"max_depth={best_config['max_depth']}, "
          f"max_features={best_config['max_features']}")
    print(f"  Validation Accuracy: {best_val_accuracy:.4f}")
    
    return best_config, best_val_accuracy, results


def cross_validate_rf(X, y, config, k_folds=10, random_state=RNG):
    """Perform k-fold cross-validation on training set with given RF configuration.
    Custom implementation - does not use sklearn's cross_val_score.
    
    Args:
        X, y: Training data
        config: RandomForest configuration dict
        k_folds: Number of folds (default 10)
        random_state: RNG object for shuffling folds (default: global RNG)
        
    Returns:
        dict: Contains fold-wise accuracies and summary statistics
    """
    fold_size = len(X) // k_folds
    indices = np.arange(len(X))
    
    rng = random_state
    rng.shuffle(indices)
    
    fold_train_accuracies = []
    fold_test_accuracies = []
    fold_details = []
    
    print("\nStep 2: 10-Fold Cross-Validation on training set with best configuration...")
    print("-" * 70)
    
    for fold in range(k_folds):
        # Create train/test split for this fold
        test_start = fold * fold_size
        test_end = test_start + fold_size if fold < k_folds - 1 else len(X)
        
        test_indices = indices[test_start:test_end]
        train_indices = np.concatenate([indices[:test_start], indices[test_end:]])
        
        X_train_fold = X[train_indices]
        y_train_fold = y[train_indices]
        X_test_fold = X[test_indices]
        y_test_fold = y[test_indices]
        
        # Train model (use integer random_state for sklearn)
        rf = RandomForestClassifier(random_state=42, **config)
        rf.fit(X_train_fold, y_train_fold)
        
        # Evaluate on train and test
        y_pred_train = rf.predict(X_train_fold)
        y_pred_test = rf.predict(X_test_fold)
        
        train_accuracy = accuracy_score(y_train_fold, y_pred_train)
        test_accuracy = accuracy_score(y_test_fold, y_pred_test)
        
        fold_train_accuracies.append(train_accuracy)
        fold_test_accuracies.append(test_accuracy)
        fold_details.append({
            "fold": fold + 1,
            "train_acc": train_accuracy,
            "test_acc": test_accuracy,
            "gap": train_accuracy - test_accuracy,
        })
        
        print(f"Fold {fold+1:2d}: Train Acc = {train_accuracy:.4f}, Test Acc = {test_accuracy:.4f}, "
              f"Gap = {train_accuracy - test_accuracy:.4f}")
    
    # Compute summary statistics
    mean_train_acc = np.mean(fold_train_accuracies)
    std_train_acc = np.std(fold_train_accuracies)
    mean_test_acc = np.mean(fold_test_accuracies)
    std_test_acc = np.std(fold_test_accuracies)
    generalization_error = mean_train_acc - mean_test_acc
    
    return {
        "fold_train_accuracies": fold_train_accuracies,
        "fold_test_accuracies": fold_test_accuracies,
        "fold_details": fold_details,
        "mean_train_acc": mean_train_acc,
        "std_train_acc": std_train_acc,
        "mean_test_acc": mean_test_acc,
        "std_test_acc": std_test_acc,
        "generalization_error": generalization_error,
    }


def define_nn_param_grid():
    """Define parameter grid for neural network (hidden layer sizes)."""
    return [
        {"hidden_layer_sizes": (50,)},
        {"hidden_layer_sizes": (100,)},
        {"hidden_layer_sizes": (150,)},
        {"hidden_layer_sizes": (200,)},
        {"hidden_layer_sizes": (50, 50)},
        {"hidden_layer_sizes": (100, 50)},
        {"hidden_layer_sizes": (100, 100)},
    ]


def tune_nn_on_validation(X_train, y_train, X_val, y_val, param_grid):
    """Evaluate Neural Network configurations on validation set and return best configuration.
    Includes feature scaling required for neural networks.
    
    Args:
        X_train, y_train: Training set
        X_val, y_val: Validation set
        param_grid: List of configuration dicts to evaluate
        
    Returns:
        tuple: (best_config, best_val_accuracy, results_list)
    """
    best_config = None
    best_val_accuracy = -1
    results = []
    
    print("\nStep 1: Evaluating Neural Network configurations on validation set...")
    print("(Note: Features are scaled on training set, applied to validation)")
    print("-" * 70)
    
    # Scale features (fit scaler on training set)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    for i, config in enumerate(param_grid, 1):
        nn = MLPClassifier(
            hidden_layer_sizes=config["hidden_layer_sizes"],
            max_iter=500,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1,
        )
        nn.fit(X_train_scaled, y_train)
        y_pred_val = nn.predict(X_val_scaled)
        val_accuracy = accuracy_score(y_val, y_pred_val)
        
        results.append({
            "config_id": i,
            "config": config,
            "val_accuracy": val_accuracy
        })
        
        print(f"Config {i}: hidden_layers={config['hidden_layer_sizes']} => Val Accuracy: {val_accuracy:.4f}")
        
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_config = config
    
    print(f"\nBest configuration on validation set:")
    print(f"  hidden_layer_sizes={best_config['hidden_layer_sizes']}")
    print(f"  Validation Accuracy: {best_val_accuracy:.4f}")
    
    return best_config, best_val_accuracy, results


def cross_validate_nn(X, y, config, k_folds=10, random_state=RNG):
    """Perform k-fold cross-validation on training set with given NN configuration.
    
    Custom implementation - does not use sklearn's cross_val_score.
    Includes feature scaling required for neural networks.
    
    Args:
        X, y: Training data
        config: Neural Network configuration dict
        k_folds: Number of folds (default 10)
        random_state: RNG object for shuffling folds (default: global RNG)
        
    Returns:
        dict: Contains fold-wise accuracies and summary statistics
    """
    fold_size = len(X) // k_folds
    indices = np.arange(len(X))
    
    rng = random_state
    rng.shuffle(indices)
    
    fold_train_accuracies = []
    fold_test_accuracies = []
    fold_details = []
    
    print("\nStep 2: 10-Fold Cross-Validation on training set with best configuration...")
    print("(Note: Features are scaled per fold, as is standard practice)")
    print("-" * 70)
    
    for fold in range(k_folds):
        # Create train/test split for this fold
        test_start = fold * fold_size
        test_end = test_start + fold_size if fold < k_folds - 1 else len(X)
        
        test_indices = indices[test_start:test_end]
        train_indices = np.concatenate([indices[:test_start], indices[test_end:]])
        
        X_train_fold = X[train_indices]
        y_train_fold = y[train_indices]
        X_test_fold = X[test_indices]
        y_test_fold = y[test_indices]
        
        # Scale features (fit scaler on training fold, apply to test fold)
        scaler = StandardScaler()
        X_train_fold_scaled = scaler.fit_transform(X_train_fold)
        X_test_fold_scaled = scaler.transform(X_test_fold)
        
        # Train model on scaled data (use integer random_state for sklearn)
        nn = MLPClassifier(
            hidden_layer_sizes=config["hidden_layer_sizes"],
            max_iter=500,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1,
        )
        nn.fit(X_train_fold_scaled, y_train_fold)
        
        # Evaluate on train and test
        y_pred_train = nn.predict(X_train_fold_scaled)
        y_pred_test = nn.predict(X_test_fold_scaled)
        
        train_accuracy = accuracy_score(y_train_fold, y_pred_train)
        test_accuracy = accuracy_score(y_test_fold, y_pred_test)
        
        fold_train_accuracies.append(train_accuracy)
        fold_test_accuracies.append(test_accuracy)
        fold_details.append({
            "fold": fold + 1,
            "train_acc": train_accuracy,
            "test_acc": test_accuracy,
            "gap": train_accuracy - test_accuracy,
        })
        
        print(f"Fold {fold+1:2d}: Train Acc = {train_accuracy:.4f}, Test Acc = {test_accuracy:.4f}, "
              f"Gap = {train_accuracy - test_accuracy:.4f}")
    
    # Compute summary statistics
    mean_train_acc = np.mean(fold_train_accuracies)
    std_train_acc = np.std(fold_train_accuracies)
    mean_test_acc = np.mean(fold_test_accuracies)
    std_test_acc = np.std(fold_test_accuracies)
    generalization_error = mean_train_acc - mean_test_acc
    
    return {
        "fold_train_accuracies": fold_train_accuracies,
        "fold_test_accuracies": fold_test_accuracies,
        "fold_details": fold_details,
        "mean_train_acc": mean_train_acc,
        "std_train_acc": std_train_acc,
        "mean_test_acc": mean_test_acc,
        "std_test_acc": std_test_acc,
        "generalization_error": generalization_error,
    }


def print_cv_results(cv_results):
    """Print formatted cross-validation results summary."""
    print("\n" + "-" * 70)
    print("10-Fold CV Summary:")
    print("-" * 70)
    print(f"Training Accuracy:   {cv_results['mean_train_acc']:.4f} ± {cv_results['std_train_acc']:.4f}")
    print(f"Test Accuracy (CV):  {cv_results['mean_test_acc']:.4f} ± {cv_results['std_test_acc']:.4f}")
    print(f"Generalization Error (Train - Test): {cv_results['generalization_error']:.4f}")
    print("=" * 70)


def analyze_learning_curve(X_train, y_train, X_val, y_val, model_config, model_type="rf", 
                           train_sizes=None, random_state=RNG):
    """Analyze learning curve: training and validation error as a function of training set size.
    
    Args:
        X_train, y_train: Full training set
        X_val, y_val: Validation set (used for test error estimation)
        model_config: Model configuration dict
        model_type: "rf" for Random Forest or "nn" for Neural Network
        train_sizes: List of training set sizes to evaluate (default: [10,20,30,...,1000])
        random_state: RNG object for reproducibility
        
    Returns:
        dict: Contains train/test accuracies at each training set size
    """
    if train_sizes is None:
        train_sizes = list(range(10, 1010, 10))
    
    train_accuracies = []
    test_accuracies = []
    
    print(f"\nLearning Curve Analysis for {model_type.upper()}:")
    print("-" * 70)
    print("Training Set Size | Train Accuracy | Validation Accuracy")
    print("-" * 70)
    
    for i, n_train in enumerate(train_sizes):
        # Sample n_train samples from training set
        indices = np.arange(len(X_train))
        random_state.shuffle(indices)
        train_indices = indices[:n_train]
        
        X_train_subset = X_train[train_indices]
        y_train_subset = y_train[train_indices]
        
        if model_type.lower() == "rf":
            # Random Forest
            model = RandomForestClassifier(random_state=42, **model_config)
            model.fit(X_train_subset, y_train_subset)
            y_pred_train = model.predict(X_train_subset)
            y_pred_val = model.predict(X_val)
            
        elif model_type.lower() == "nn":
            # Neural Network (with feature scaling, no early stopping for learning curves)
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train_subset)
            X_val_scaled = scaler.transform(X_val)
            
            model = MLPClassifier(
                hidden_layer_sizes=model_config["hidden_layer_sizes"],
                max_iter=1000,
                random_state=42,
                early_stopping=False,  # Disable for learning curves
            )
            model.fit(X_train_scaled, y_train_subset)
            y_pred_train = model.predict(X_train_scaled)
            y_pred_val = model.predict(X_val_scaled)
        
        train_acc = accuracy_score(y_train_subset, y_pred_train)
        test_acc = accuracy_score(y_val, y_pred_val)
        
        train_accuracies.append(train_acc)
        test_accuracies.append(test_acc)
        
        if n_train % 100 == 0 or n_train <= 30:
            print(f"{n_train:17d} | {train_acc:14.4f} | {test_acc:19.4f}")
    
    print("-" * 70)
    
    return {
        "train_sizes": train_sizes,
        "train_accuracies": train_accuracies,
        "test_accuracies": test_accuracies,
    }


def plot_learning_curves(rf_results, nn_results, output_dir=None):
    """Plot learning curves for both Random Forest and Neural Network.
    
    Args:
        rf_results: Results dict from analyze_learning_curve for RF
        nn_results: Results dict from analyze_learning_curve for NN
        output_dir: Directory to save the figure (optional). If not specified, uses default plots directory.
    """
    plt.switch_backend('Agg')  # Use non-interactive backend
    
    # Create output directory if not specified
    if output_dir is None:
        output_dir = Path(__file__).resolve().parent / "plot_outputs"
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate filename with timestamp for tracking different runs
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"learning_curves_{timestamp}.png"
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Random Forest
    axes[0].plot(rf_results["train_sizes"], rf_results["train_accuracies"], 
                 'o-', label='Training Accuracy', linewidth=2, markersize=4)
    axes[0].plot(rf_results["train_sizes"], rf_results["test_accuracies"], 
                 's-', label='Validation Accuracy', linewidth=2, markersize=4)
    axes[0].set_xlabel('Training Set Size (N)', fontsize=12)
    axes[0].set_ylabel('Accuracy', fontsize=12)
    axes[0].set_title('Random Forest - Learning Curve', fontsize=13, fontweight='bold')
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim([0, 1.1])
    
    # Neural Network
    axes[1].plot(nn_results["train_sizes"], nn_results["train_accuracies"], 
                 'o-', label='Training Accuracy', linewidth=2, markersize=4)
    axes[1].plot(nn_results["train_sizes"], nn_results["test_accuracies"], 
                 's-', label='Validation Accuracy', linewidth=2, markersize=4)
    axes[1].set_xlabel('Training Set Size (N)', fontsize=12)
    axes[1].set_ylabel('Accuracy', fontsize=12)
    axes[1].set_title('Neural Network - Learning Curve', fontsize=13, fontweight='bold')
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim([0, 1.1])
    
    plt.tight_layout()
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nFigure saved to: {output_path}")
    
    return fig






def activate():
    """High-level orchestration function that runs the full pipeline and returns a dict of artifacts.
    This is the function you can call interactively; each step is modular and testable.
    """
    base_dir = Path(__file__).resolve().parent
    mu1, mu2, Sigma1, Sigma2 = load_true_parameters(base_dir)

    ## Q2a 
    df, X, y = generate_dataset(mu1, mu2, Sigma1, Sigma2, N_total=10_000)
    
    ## Q2b
    estimates = compute_mle_estimates(df)
    errors = compute_errors(estimates, (mu1, mu2, Sigma1, Sigma2))
    print_mle_results(estimates, errors)

    ## Q2c
    df_val, X_val, y_val = generate_dataset(mu1, mu2, Sigma1, Sigma2, N_total=2000)

    ## Q2d 
    print("=" * 70)
    print("Random Forest:")
    param_grid = define_param_grid()
    best_config, best_val_accuracy, val_results = tune_rf_on_validation(X, y, X_val, y_val, param_grid)
    cv_results = cross_validate_rf(X, y, best_config, k_folds=10, random_state=RNG)
    print_cv_results(cv_results)

    ## Q2e
    print("=" * 70)
    print("Neural Network:")
    nn_param_grid = define_nn_param_grid()
    best_nn_config, best_nn_val_accuracy, nn_val_results = tune_nn_on_validation(X, y, X_val, y_val, nn_param_grid)
    nn_cv_results = cross_validate_nn(X, y, best_nn_config, k_folds=10, random_state=RNG)
    print_cv_results(nn_cv_results)

    ## Q2f
    # learning curves for Random Forest
    rf_learning_results = analyze_learning_curve(
        X, y, X_val, y_val, 
        model_config=best_config, 
        model_type="rf",
        train_sizes=list(range(10, 1010, 10)),
        random_state=RNG)
    
    # learning curves for Neural Network
    nn_learning_results = analyze_learning_curve(
        X, y, X_val, y_val,
        model_config=best_nn_config,
        model_type="nn",
        train_sizes=list(range(10, 1010, 10)),
        random_state=RNG)
    
    # Plot learning curves
    output_dir = Path(__file__).resolve().parent / "plot_outputs"
    plot_learning_curves(rf_learning_results, nn_learning_results, output_dir=str(output_dir))

    return {
        "dataset": df,
        "X": X,
        "y": y,
        "estimates": estimates,
        "errors": errors,
        "df_val": df_val,
        "X_val": X_val,
        "y_val": y_val,
        # RF results
        "rf_param_grid": param_grid,
        "rf_best_config": best_config,
        "rf_best_val_accuracy": best_val_accuracy,
        "rf_val_results": val_results,
        "rf_cv_results": cv_results,
        # NN results
        "nn_param_grid": nn_param_grid,
        "nn_best_config": best_nn_config,
        "nn_best_val_accuracy": best_nn_val_accuracy,
        "nn_val_results": nn_val_results,
        "nn_cv_results": nn_cv_results,
        # Learning curve results
        "rf_learning_results": rf_learning_results,
        "nn_learning_results": nn_learning_results,
    }



if __name__ == "__main__":
    activate()


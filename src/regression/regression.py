from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score

from src.dataloader import load_random_gauges
from preprocess_regression import Preprocessor

# Load data
DATA_DIR = Path("../..") / "data"
gauges = load_random_gauges(DATA_DIR, n_samples=100, seed=42)

# Preprocess
preprocessor = Preprocessor(
    target_col='prec',
    temporal_split=True,
    lag_days=[],
    # lag_days=[1, 3, 7],
    rolling_windows=[]
    # rolling_windows=[3, 7]
)

train_data, val_data, test_data = preprocessor.preprocess(gauges)
X_train, y_train = train_data
X_val, y_val = val_data
X_test, y_test = test_data

print(f"Training samples: {len(X_train)}")
print(f"Features: {len(preprocessor.feature_cols)}\n")

# Try different models
models = {
    'Linear': LinearRegression(), # Good baseline
    'Ridge': Ridge(alpha=1.0), # (L2 Regularization), Handles many correlated features well
    'Lasso': Lasso(alpha=0.1) # (L1 Regularization), Narrows down to only few features
}

results = {}
feature_importances = {}   # store coefficient-based importance

for name, model in models.items():
    # Train
    model.fit(X_train, y_train)

    # Evaluate
    train_pred = model.predict(X_train)
    val_pred = model.predict(X_val)
    test_pred = model.predict(X_test)

    train_rmse = mean_squared_error(y_train, train_pred)
    val_rmse = mean_squared_error(y_val, val_pred)
    test_rmse = mean_squared_error(y_test, test_pred)
    test_r2 = r2_score(y_test, test_pred)

    print(f"{name:8} - Train: {train_rmse:.2f}, Val: {val_rmse:.2f}, Test: {test_rmse:.2f} mm, R²: {test_r2:.3f}")

    results[name] = {
        "train_rmse": train_rmse,
        "val_rmse": val_rmse,
        "test_rmse": test_rmse,
        "test_r2": test_r2,
    }

    # --------- Coefficient-based feature importance ---------
    coefs = model.coef_  # shape: (n_features,)
    fi_df = pd.DataFrame({
        "feature": preprocessor.feature_cols,
        "coef": coefs,
        "abs_coef": np.abs(coefs),
    }).sort_values("abs_coef", ascending=False)

    feature_importances[name] = fi_df

    print(f"\nTop 10 features for {name} (by |coef|):")
    print(fi_df.head(10).to_string(index=False))
    print("\n" + "-" * 60 + "\n")
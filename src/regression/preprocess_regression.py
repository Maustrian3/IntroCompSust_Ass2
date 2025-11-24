# src/preprocessing.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import torch
from typing import Dict, Tuple, Optional
from pathlib import Path


class Preprocessor:

    def __init__(
            self,
            target_col: str = 'prec',
            test_size: float = 0.15,
            val_size: float = 0.15,
            temporal_split: bool = True,
            lag_days: list[int] = [],
            rolling_windows: list[int] = [],
            random_state: int = 42
    ):
        """
        Args:
            target_col: Column to predict (default: 'prec')
            test_size: Proportion of data for test set
            val_size: Proportion of training data for validation
            temporal_split: If True, split by date; if False, random split
            lag_days: List of lag days to create (e.g., [1, 3, 7])
            rolling_windows: List of window sizes for rolling means
            random_state: Random seed
        """
        self.target_col = target_col
        self.test_size = test_size
        self.val_size = val_size
        self.temporal_split = temporal_split
        self.lag_days = lag_days
        self.rolling_windows = rolling_windows
        self.random_state = random_state

        self.feature_scaler = None
        self.target_scaler = None
        self.feature_cols = None

    def consolidate_gauges(
            self,
            gauges: Dict[str, pd.DataFrame]
    ) -> pd.DataFrame:
        """Vertically stack all gauge data with gauge_id column."""
        dfs = []
        for gauge_id, df in gauges.items():
            df = df.copy()
            df['gauge_id'] = gauge_id
            dfs.append(df)

        combined = pd.concat(dfs, ignore_index=True)

        # Ensure date column exists
        if 'date' not in combined.columns:
            combined['date'] = pd.to_datetime(
                combined[['YYYY', 'MM', 'DD']].rename(
                    columns={'YYYY': 'year', 'MM': 'month', 'DD': 'day'}
                )
            )

        return combined.sort_values(['gauge_id', 'date']).reset_index(drop=True)

    def create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create cyclical temporal features from date. Encodes that e.g. Dec-Jan is as far apart as Feb-Mar"""
        df = df.copy()

        # Extract temporal components
        df['month'] = df['date'].dt.month
        df['day_of_year'] = df['date'].dt.dayofyear

        # Cyclical encoding for month
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

        # Cyclical encoding for day of year
        df['doy_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
        df['doy_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365)

        return df

    def create_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create lagged features for key weather variables. Directly encodes feature relevance of previous days into the dataset"""
        df = df.copy()

        # Key features to create lags for
        lag_features = [
            '2m_temp_mean', '2m_dp_temp_mean', 'surf_press',
            'total_et', self.target_col
        ]

        for feature in lag_features:
            if feature not in df.columns:
                continue
            for lag in self.lag_days:
                df[f'{feature}_lag{lag}'] = df.groupby('gauge_id')[feature].shift(lag)

        return df

    def create_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create rolling mean features. Smoothing out values to reduce noise"""
        df = df.copy()

        roll_features = [
            '2m_temp_mean', 'surf_press', self.target_col
        ]

        for feature in roll_features:
            if feature not in df.columns:
                continue
            for window in self.rolling_windows:
                df[f'{feature}_roll{window}'] = (
                    df.groupby('gauge_id')[feature]
                    .rolling(window=window, min_periods=1)
                    .mean()
                    .reset_index(level=0, drop=True)
                )

        return df

    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values with forward/backward fill per gauge."""
        df = df.copy()

        # Forward fill then backward fill within each gauge
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            df[col] = df.groupby('gauge_id')[col].ffill().bfill()

        # Drop remaining NaNs (from lag features at start)
        df = df.dropna()

        return df

    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Full feature engineering pipeline."""
        df = self.create_temporal_features(df)
        df = self.create_lag_features(df)
        df = self.create_rolling_features(df)
        df = self.handle_missing_values(df)
        return df

    def split_data(
            self,
            df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split data into train/val/test sets. Avoids leakage for prediction, otherwise days in the future could be used for prediction."""
        if self.temporal_split:
            # Sort by date and split chronologically
            df = df.sort_values('date')
            n = len(df)
            test_idx = int(n * (1 - self.test_size))
            train_val = df.iloc[:test_idx]
            test = df.iloc[test_idx:]

            # Split train into train/val
            n_tv = len(train_val)
            val_idx = int(n_tv * (1 - self.val_size))
            train = train_val.iloc[:val_idx]
            val = train_val.iloc[val_idx:]
        else:
            # Random split
            train_val, test = train_test_split(
                df, test_size=self.test_size, random_state=self.random_state
            )
            train, val = train_test_split(
                train_val, test_size=self.val_size, random_state=self.random_state
            )

        return train, val, test

    def fit_scalers(self, train_df: pd.DataFrame) -> None:
        """Fit scalers on training data only."""
        # Select numeric features (exclude date, gauge_id, target)
        exclude_cols = ['date', 'gauge_id', 'YYYY', 'MM', 'DD', 'DOY', self.target_col]
        self.feature_cols = [
            col for col in train_df.columns
            if col not in exclude_cols and train_df[col].dtype in [np.float64, np.int64]
        ]

        # Fit feature scaler
        self.feature_scaler = StandardScaler()
        self.feature_scaler.fit(train_df[self.feature_cols])

        # Fit target scaler (optional, but can help)
        self.target_scaler = StandardScaler()
        self.target_scaler.fit(train_df[[self.target_col]])

    def transform_data(
            self,
            df: pd.DataFrame,
            scale_target: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Transform data using fitted scalers."""
        X = self.feature_scaler.transform(df[self.feature_cols])

        if scale_target:
            y = self.target_scaler.transform(df[[self.target_col]]).flatten()
        else:
            y = df[self.target_col].values

        return X, y

    def inverse_transform_target(self, y_scaled: np.ndarray) -> np.ndarray:
        """Inverse transform predictions back to original scale."""
        return self.target_scaler.inverse_transform(
            y_scaled.reshape(-1, 1)
        ).flatten()

    def preprocess(
            self,
            gauges: Dict[str, pd.DataFrame]
    ) -> Tuple[
        Tuple[np.ndarray, np.ndarray],
        Tuple[np.ndarray, np.ndarray],
        Tuple[np.ndarray, np.ndarray]
    ]:
        """
        Complete preprocessing pipeline.

        Returns:
            (X_train, y_train), (X_val, y_val), (X_test, y_test)
        """
        # Consolidate data
        df = self.consolidate_gauges(gauges)
        print(f"Consolidated data shape: {df.shape}")

        # Engineer features
        df = self.prepare_features(df)
        print(f"After feature engineering: {df.shape}")

        # Split data
        train_df, val_df, test_df = self.split_data(df)
        print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

        # Fit scalers on training data
        self.fit_scalers(train_df)
        print(f"Number of features: {len(self.feature_cols)}")

        # Transform all sets
        X_train, y_train = self.transform_data(train_df)
        X_val, y_val = self.transform_data(val_df)
        X_test, y_test = self.transform_data(test_df)

        return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def to_tensors(
        train_data: Tuple[np.ndarray, np.ndarray],
        val_data: Tuple[np.ndarray, np.ndarray],
        test_data: Tuple[np.ndarray, np.ndarray]
) -> Tuple[
    Tuple[torch.Tensor, torch.Tensor],
    Tuple[torch.Tensor, torch.Tensor],
    Tuple[torch.Tensor, torch.Tensor]
]:
    """Convert numpy arrays to PyTorch tensors."""

    X_train, y_train = train_data
    X_val, y_val = val_data
    X_test, y_test = test_data

    return (
        (torch.FloatTensor(X_train), torch.FloatTensor(y_train)),
        (torch.FloatTensor(X_val), torch.FloatTensor(y_val)),
        (torch.FloatTensor(X_test), torch.FloatTensor(y_test))
    )


if __name__ == "__main__":
    from src.dataloader import load_random_gauges

    # Load data
    DATA_DIR = Path("../..") / "data"
    gauges = load_random_gauges(DATA_DIR, n_samples=100, seed=42)

    # Preprocess
    preprocessor = Preprocessor(
        target_col='prec',
        temporal_split=True,
        lag_days=[1, 3, 7],
        rolling_windows=[3, 7]
    )

    train_data, val_data, test_data = preprocessor.preprocess(gauges)

    # Convert to tensors
    train_tensors, val_tensors, test_tensors = to_tensors(
        train_data, val_data, test_data
    )

    X_train, y_train = train_tensors
    X_val, y_val = val_tensors
    X_test, y_test = test_tensors

    print(f"\nFeature columns ({len(preprocessor.feature_cols)}):")
    print(preprocessor.feature_cols)

    print(f"\nTensor shapes:")
    print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"X_val: {X_val.shape}, y_val: {y_val.shape}")
    print(f"X_test: {X_test.shape}, y_test: {y_test.shape}")

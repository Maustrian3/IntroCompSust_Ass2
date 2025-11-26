# preprocessing_lstm.py
from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib


class SequencePreprocessor:
    """Preprocess data for LSTM/GRU models and save artifacts."""

    def __init__(
        self,
        target_col: str = "prec",
        sequence_length: int = 14,  # Use past 14 days to predict tomorrow
        test_size: float = 0.1,
        val_size: float = 0.1,
        random_state: int = 42,
    ):
        """
        Args:
            target_col: Column to predict.
            sequence_length: Number of past days to use as input.
            test_size: Proportion for test set.
            val_size: Proportion of training for validation.
        """
        self.target_col = target_col
        self.sequence_length = sequence_length
        self.test_size = test_size
        self.val_size = val_size
        self.random_state = random_state

        self.feature_scaler: StandardScaler | None = None
        self.target_scaler: StandardScaler | None = None
        self.feature_cols: List[str] | None = None

    def consolidate_gauges(self, gauges: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Stack all gauge data."""
        dfs = []
        for gauge_id, df in gauges.items():
            df = df.copy()
            df["gauge_id"] = gauge_id
            dfs.append(df)

        combined = pd.concat(dfs, ignore_index=True)

        if "date" not in combined.columns:
            combined["date"] = pd.to_datetime(
                combined[["YYYY", "MM", "DD"]].rename(
                    columns={"YYYY": "year", "MM": "month", "DD": "day"}
                )
            )

        return combined.sort_values(["gauge_id", "date"]).reset_index(drop=True)

    def create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add cyclical temporal features."""
        df = df.copy()

        df["month"] = df["date"].dt.month
        df["day_of_year"] = df["date"].dt.dayofyear

        # Cyclical encoding
        df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
        df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
        df["doy_sin"] = np.sin(2 * np.pi * df["day_of_year"] / 365)
        df["doy_cos"] = np.cos(2 * np.pi * df["day_of_year"] / 365)

        return df

    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Forward/backward fill missing values per gauge."""
        df = df.copy()
        numeric_cols = df.select_dtypes(include=[np.number]).columns

        for col in numeric_cols:
            df[col] = df.groupby("gauge_id")[col].ffill().bfill()

        df = df.dropna()
        return df

    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Feature engineering pipeline (no lag features needed!)."""
        df = self.create_temporal_features(df)
        df = self.handle_missing_values(df)
        return df

    def create_sequences(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Create sequences for LSTM input.

        Returns:
            X: (num_sequences, sequence_length, num_features)
            y: (num_sequences,)
            dates: (num_sequences,) for tracking
        """
        sequences: list[np.ndarray] = []
        targets: list[float] = []
        dates: list[pd.Timestamp] = []

        # Group by gauge to avoid mixing data across sensors
        for gauge_id, group in df.groupby("gauge_id"):
            group = group.sort_values("date").reset_index(drop=True)

            # Create sequences
            for i in range(len(group) - self.sequence_length):
                # Input: past sequence_length days
                seq = group.iloc[i : i + self.sequence_length][feature_cols].values
                # Target: precipitation on day sequence_length + 1
                target = group.iloc[i + self.sequence_length][self.target_col] ## TODO is  +1 actually happening?
                date = group.iloc[i + self.sequence_length]["date"]

                sequences.append(seq)
                targets.append(target)
                dates.append(date)

        X = np.array(sequences)  # (num_sequences, sequence_length, num_features)
        y = np.array(targets)  # (num_sequences,)
        dates = np.array(dates)

        return X, y, dates

    def split_by_date(
        self,
        X: np.ndarray,
        y: np.ndarray,
        dates: np.ndarray,
    ) -> Tuple[
        Tuple[np.ndarray, np.ndarray],
        Tuple[np.ndarray, np.ndarray],
        Tuple[np.ndarray, np.ndarray],
    ]:
        """Temporal split by date."""
        sort_idx = np.argsort(dates)
        X = X[sort_idx]
        y = y[sort_idx]
        dates = dates[sort_idx]

        n = len(X)
        test_idx = int(n * (1 - self.test_size))
        val_idx = int(test_idx * (1 - self.val_size))

        X_train, y_train = X[:val_idx], y[:val_idx]
        X_val, y_val = X[val_idx:test_idx], y[val_idx:test_idx]
        X_test, y_test = X[test_idx:], y[test_idx:]

        return (X_train, y_train), (X_val, y_val), (X_test, y_test)

    def fit_scalers(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """Fit scalers on training data."""
        # Reshape for scaling: (num_sequences * sequence_length, num_features)
        X_reshaped = X_train.reshape(-1, X_train.shape[-1])

        self.feature_scaler = StandardScaler()
        self.feature_scaler.fit(X_reshaped)

        self.target_scaler = StandardScaler()
        self.target_scaler.fit(y_train.reshape(-1, 1))

    def scale_sequences(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Scale sequences and targets."""
        if self.feature_scaler is None or self.target_scaler is None:
            raise RuntimeError("Scalers are not fitted yet. Call fit_scalers first.")

        # Scale features
        original_shape = X.shape
        X_reshaped = X.reshape(-1, X.shape[-1])
        X_scaled = self.feature_scaler.transform(X_reshaped)
        X_scaled = X_scaled.reshape(original_shape)

        # Scale target
        y_scaled = self.target_scaler.transform(y.reshape(-1, 1)).flatten()

        return X_scaled, y_scaled

    def preprocess(
        self,
        gauges: Dict[str, pd.DataFrame],
    ) -> Tuple[
        Tuple[np.ndarray, np.ndarray],
        Tuple[np.ndarray, np.ndarray],
        Tuple[np.ndarray, np.ndarray],
    ]:
        """Complete preprocessing pipeline."""
        # Consolidate
        df = self.consolidate_gauges(gauges)
        print(f"Consolidated data: {df.shape}")

        # Feature engineering
        df = self.prepare_features(df)
        print(f"After feature engineering: {df.shape}")

        # Select features (no target, no identifiers)
        exclude_cols = [
            "date",
            "gauge_id",
            "YYYY",
            "MM",
            "DD",
            "DOY",
            "month",
            "day_of_year",
            self.target_col,
        ]
        self.feature_cols = [
            col
            for col in df.columns
            if col not in exclude_cols and df[col].dtype in [np.float64, np.int64]
        ]

        print(f"Features for sequences: {len(self.feature_cols)}")

        # Create sequences
        X, y, dates = self.create_sequences(df, self.feature_cols)
        print(f"Created {len(X)} sequences of shape {X.shape[1:]}")

        # Split temporally
        train_data, val_data, test_data = self.split_by_date(X, y, dates)
        X_train, y_train = train_data
        X_val, y_val = val_data
        X_test, y_test = test_data

        print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

        # Fit and apply scalers
        self.fit_scalers(X_train, y_train)

        X_train, y_train = self.scale_sequences(X_train, y_train)
        X_val, y_val = self.scale_sequences(X_val, y_val)
        X_test, y_test = self.scale_sequences(X_test, y_test)

        return (X_train, y_train), (X_val, y_val), (X_test, y_test)

    def save_artifacts(
        self,
        train_data: Tuple[np.ndarray, np.ndarray],
        val_data: Tuple[np.ndarray, np.ndarray],
        test_data: Tuple[np.ndarray, np.ndarray],
        out_dir: str | Path = "preprocessed",
    ) -> None:
        """
        Save scaled sequences + scalers so they can be loaded in Colab.

        Creates:
          - <out_dir>/lstm_dataset.npz
          - <out_dir>/scalers.pkl
        """
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        X_train, y_train = train_data
        X_val, y_val = val_data
        X_test, y_test = test_data

        dataset_path = out_dir / "lstm_dataset.npz"
        np.savez_compressed(
            dataset_path,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            X_test=X_test,
            y_test=y_test,
            seq_len=self.sequence_length,
            feature_cols=np.array(self.feature_cols, dtype=object),
            target_col=self.target_col,
        )

        scalers_path = out_dir / "scalers.pkl"
        joblib.dump(
            {
                "feature_scaler": self.feature_scaler,
                "target_scaler": self.target_scaler,
            },
            scalers_path,
        )

        print(f"Saved dataset to: {dataset_path}")
        print(f"Saved scalers  to: {scalers_path}")


if __name__ == "__main__":
    from dataloader import load_random_gauges

    DATA_DIR = Path("..") / "data"
    gauges = load_random_gauges(DATA_DIR, n_samples=100, seed=42) # TODO re-preprocess everything with 100 samples

    preprocessor = SequencePreprocessor(
        target_col="prec",
        sequence_length=30, # Use last 30 days for prediction
    )

    train_data, val_data, test_data = preprocessor.preprocess(gauges)

    preprocessor.save_artifacts(train_data, val_data, test_data, out_dir="preprocessed_30")

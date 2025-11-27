# preprocessing_lstm.py
from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib
from tqdm.auto import tqdm


class SequencePreprocessor:
    """Preprocess data for LSTM/GRU models and save artifacts."""

    def __init__(
        self,
        target_col: str = "prec",
        sequence_length: int = 14,  # Use past 14 days to predict tomorrow
        test_size: float = 0.1,
        val_size: float = 0.1,
        lag_days: list[int] = [],
        rolling_windows: list[int] = [],
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
        self.lag_days = lag_days
        self.rolling_windows = rolling_windows
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

        # Key features to create rolling windows for
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
        """Forward/backward fill missing values per gauge."""
        df = df.copy()
        numeric_cols = df.select_dtypes(include=[np.number]).columns

        for col in numeric_cols:
            df[col] = df.groupby("gauge_id")[col].ffill().bfill()

        df = df.dropna()
        return df

    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Feature engineering pipeline."""
        df = self.create_temporal_features(df)
        df = self.create_lag_features(df)
        df = self.create_rolling_features(df)
        df = self.handle_missing_values(df)

        # Cast numeric columns to float32 to reduce memory
        num_cols = df.select_dtypes(include=[np.number]).columns
        df[num_cols] = df[num_cols].astype(np.float32)
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

            feature_matrix = group[feature_cols].values
            target_array = group[self.target_col].values
            date_array = group["date"].values

            # Create sequences
            for i in range(len(feature_matrix) - self.sequence_length):
                # Input: past sequence_length days
                seq = feature_matrix[i: i + self.sequence_length]
                # Target: precipitation on day sequence_length + 1
                target = target_array[i + self.sequence_length]
                date = date_array[i + self.sequence_length]

                seq_dates = date_array[i: i + self.sequence_length]
                target_date = date_array[i + self.sequence_length]

                sequences.append(seq)
                targets.append(target)
                dates.append(date)

        X = np.array(sequences, dtype=np.float32)  # (num_sequences, sequence_length, num_features)
        y = np.array(targets, dtype=np.float32)  # (num_sequences,)
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
        """Fit scalers on training data in batches."""
        n_seq, seq_len, n_feat = X_train.shape

        max_rows_per_batch = 300_000
        batch_size = max(1, max_rows_per_batch // seq_len)

        print(
            f"[fit_scalers] n_seq={n_seq}, seq_len={seq_len}, "
            f"n_feat={n_feat}, batch_size={batch_size}"
        )

        # Feature scaler with in-place behaviour
        self.feature_scaler = StandardScaler(copy=False)

        # Batched partial_fit over views of X_train
        for start in range(0, n_seq, batch_size):
            end = min(start + batch_size, n_seq)
            batch_view = X_train[start:end].reshape(-1, n_feat)  # view, no copy
            self.feature_scaler.partial_fit(batch_view)

        # Target scaler
        self.target_scaler = StandardScaler(copy=False)
        self.target_scaler.fit(y_train.reshape(-1, 1))

    def scale_sequences(
            self,
            X: np.ndarray,
            y: np.ndarray,
            max_rows_per_batch: int = 300_000,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Scale sequences and targets in-place in batches."""
        if self.feature_scaler is None or self.target_scaler is None:
            raise RuntimeError("Scalers are not fitted yet. Call fit_scalers first.")

        n_seq, seq_len, n_feat = X.shape
        batch_size = max(1, max_rows_per_batch // seq_len)

        # Scale X in-place, batch by batch
        for start in range(0, n_seq, batch_size):
            end = min(start + batch_size, n_seq)
            batch_view = X[start:end].reshape(-1, n_feat)  # view into X
            # copy=False is controlled by the scaler itself
            self.feature_scaler.transform(batch_view)

        # Scale y in-place-ish (this one is tiny compared to X)
        y_view = y.reshape(-1, 1)
        self.target_scaler.transform(y_view)
        y_scaled = y_view.ravel()

        return X, y_scaled

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
            "day_of_year"
        ]
        self.feature_cols = [
            col
            for col in df.columns
            if col not in exclude_cols and df[col].dtype.kind in ("f", "i")
        ]

        print(f"{len(self.feature_cols)} Features for sequences: {self.feature_cols}")

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
    N_SAMPLES = 100
    SEED = 42
    SEQ_LEN = 3 # Use last SEQ_LEN days for prediction

    LAG_DAYS = [1, 3, 7]
    ROLLING_WINDOWS = [3, 7]

    gauges = load_random_gauges(DATA_DIR, n_samples=N_SAMPLES, seed=SEED) # TODO re-preprocess everything with 100 samples

    preprocessor = SequencePreprocessor(
        target_col="prec",
        lag_days=LAG_DAYS,
        rolling_windows=ROLLING_WINDOWS,
        sequence_length=SEQ_LEN,
    )

    train_data, val_data, test_data = preprocessor.preprocess(gauges)

    preprocessor.save_artifacts(train_data, val_data, test_data, out_dir=f"preprocessed_{SEQ_LEN}")

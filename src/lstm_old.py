# src/lstm_preprocessing.py
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from typing import Dict, Tuple, List


class SequencePreprocessor:
    """Preprocess data for LSTM/GRU models."""

    def __init__(
            self,
            target_col: str = 'prec',
            sequence_length: int = 14,  # Use past 14 days to predict tomorrow
            test_size: float = 0.2,
            val_size: float = 0.1,
            random_state: int = 42
    ):
        """
        Args:
            target_col: Column to predict
            sequence_length: Number of past days to use as input
            test_size: Proportion for test set
            val_size: Proportion of training for validation
        """
        self.target_col = target_col
        self.sequence_length = sequence_length
        self.test_size = test_size
        self.val_size = val_size
        self.random_state = random_state

        self.feature_scaler = None
        self.target_scaler = None
        self.feature_cols = None

    def consolidate_gauges(self, gauges: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Stack all gauge data."""
        dfs = []
        for gauge_id, df in gauges.items():
            df = df.copy()
            df['gauge_id'] = gauge_id
            dfs.append(df)

        combined = pd.concat(dfs, ignore_index=True)

        if 'date' not in combined.columns:
            combined['date'] = pd.to_datetime(
                combined[['YYYY', 'MM', 'DD']].rename(
                    columns={'YYYY': 'year', 'MM': 'month', 'DD': 'day'}
                )
            )

        return combined.sort_values(['gauge_id', 'date']).reset_index(drop=True)

    def create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add cyclical temporal features."""
        df = df.copy()

        df['month'] = df['date'].dt.month
        df['day_of_year'] = df['date'].dt.dayofyear

        # Cyclical encoding
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        df['doy_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
        df['doy_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365)

        return df

    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Forward/backward fill missing values per gauge."""
        df = df.copy()
        numeric_cols = df.select_dtypes(include=[np.number]).columns

        for col in numeric_cols:
            df[col] = df.groupby('gauge_id')[col].ffill().bfill()

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
            feature_cols: List[str]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Create sequences for LSTM input.

        Returns:
            X: (num_sequences, sequence_length, num_features)
            y: (num_sequences,)
            dates: (num_sequences,) for tracking
        """
        sequences = []
        targets = []
        dates = []

        # Group by gauge to avoid mixing data across sensors
        for gauge_id, group in df.groupby('gauge_id'):
            group = group.sort_values('date').reset_index(drop=True)

            # Create sequences
            for i in range(len(group) - self.sequence_length):
                # Input: past sequence_length days
                seq = group.iloc[i:i + self.sequence_length][feature_cols].values
                # Target: precipitation on day sequence_length + 1
                target = group.iloc[i + self.sequence_length][self.target_col]
                date = group.iloc[i + self.sequence_length]['date']

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
            dates: np.ndarray
    ) -> Tuple[
        Tuple[np.ndarray, np.ndarray],
        Tuple[np.ndarray, np.ndarray],
        Tuple[np.ndarray, np.ndarray]
    ]:
        """Temporal split by date."""
        # Sort by date
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
            y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Scale sequences and targets."""
        # Scale features
        original_shape = X.shape
        X_reshaped = X.reshape(-1, X.shape[-1])
        X_scaled = self.feature_scaler.transform(X_reshaped)
        X_scaled = X_scaled.reshape(original_shape)

        # Scale target
        y_scaled = self.target_scaler.transform(y.reshape(-1, 1)).flatten()

        return X_scaled, y_scaled

    def inverse_transform_target(self, y_scaled: np.ndarray) -> np.ndarray:
        """Inverse transform predictions."""
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
        """Complete preprocessing pipeline."""
        # Consolidate
        df = self.consolidate_gauges(gauges)
        print(f"Consolidated data: {df.shape}")

        # Feature engineering
        df = self.prepare_features(df)
        print(f"After feature engineering: {df.shape}")

        # Select features (no target, no identifiers)
        exclude_cols = ['date', 'gauge_id', 'YYYY', 'MM', 'DD', 'DOY',
                        'month', 'day_of_year', self.target_col]
        self.feature_cols = [
            col for col in df.columns
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


class SequenceDataset(Dataset):
    """PyTorch Dataset for sequences."""

    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class LSTMPredictor(nn.Module):
    """LSTM model for precipitation prediction."""

    def __init__(
            self,
            input_dim: int,
            hidden_dim: int = 64,
            num_layers: int = 2,
            dropout: float = 0.2
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        # x: (batch, sequence_length, input_dim)
        lstm_out, _ = self.lstm(x)
        # Use last timestep
        last_output = lstm_out[:, -1, :]
        # Predict
        output = self.fc(last_output)
        return output.squeeze()


class GRUPredictor(nn.Module):
    """GRU model (lighter alternative to LSTM)."""

    def __init__(
            self,
            input_dim: int,
            hidden_dim: int = 64,
            num_layers: int = 2,
            dropout: float = 0.2
    ):
        super().__init__()

        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        gru_out, _ = self.gru(x)
        last_output = gru_out[:, -1, :]
        output = self.fc(last_output)
        return output.squeeze()


def train_model(
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 50,
        lr: float = 0.001,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
):
    """Train the model."""
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_loss = float('inf')

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            predictions = model(X_batch)
            loss = criterion(predictions, y_batch)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                predictions = model(X_batch)
                loss = criterion(predictions, y_batch)
                val_loss += loss.item()

        val_loss /= len(val_loader)

        print(f"Epoch {epoch + 1}/{epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pt')

    # Load best model
    model.load_state_dict(torch.load('best_model.pt'))
    return model


def evaluate_model(
        model: nn.Module,
        test_loader: DataLoader,
        preprocessor: SequencePreprocessor,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
):
    """Evaluate model on test set."""
    model = model.to(device)
    model.eval()

    predictions = []
    actuals = []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            pred = model(X_batch).cpu().numpy()
            predictions.extend(pred)
            actuals.extend(y_batch.numpy())

    predictions = np.array(predictions)
    actuals = np.array(actuals)

    # Inverse transform to get actual mm values
    predictions = preprocessor.inverse_transform_target(predictions)
    actuals = preprocessor.inverse_transform_target(actuals)

    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

    rmse = mean_squared_error(actuals, predictions)
    mae = mean_absolute_error(actuals, predictions)
    r2 = r2_score(actuals, predictions)

    print(f"\nTest Results:")
    print(f"RMSE: {rmse:.3f} mm")
    print(f"MAE:  {mae:.3f} mm")
    print(f"R²:   {r2:.4f}")

    return predictions, actuals


# Example usage
if __name__ == "__main__":
    from dataloader import load_random_gauges

    # Load data
    DATA_DIR = Path("..") / "data"
    gauges = load_random_gauges(DATA_DIR, n_samples=50, seed=42)

    # Preprocess for sequences
    preprocessor = SequencePreprocessor(
        target_col='prec',
        sequence_length=14  # Use 14 days to predict day 15
    )

    train_data, val_data, test_data = preprocessor.preprocess(gauges)

    # Create datasets and loaders
    train_dataset = SequenceDataset(*train_data)
    val_dataset = SequenceDataset(*val_data)
    test_dataset = SequenceDataset(*test_data)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # Create model
    input_dim = train_data[0].shape[2]  # num_features
    model = LSTMPredictor(input_dim=input_dim, hidden_dim=64, num_layers=2)

    print(f"\nModel: LSTM with {input_dim} input features")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")

    # Train
    model = train_model(model, train_loader, val_loader, epochs=50, lr=0.001)

    # Evaluate
    predictions, actuals = evaluate_model(model, test_loader, preprocessor)
# lstm.py
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import joblib


def load_preprocessed_artifacts(
    base_dir: str | Path = ".",
    dataset_file: str = "lstm_dataset.npz",
    scaler_file: str = "scalers.pkl",
):
    """
    Load X/y splits + scalers saved by preprocessing_lstm.py.
    """
    base_dir = Path(base_dir)

    data = np.load(base_dir / dataset_file, allow_pickle=True)

    X_train = data["X_train"]
    y_train = data["y_train"]
    X_val = data["X_val"]
    y_val = data["y_val"]
    X_test = data["X_test"]
    y_test = data["y_test"]

    seq_len = int(data["seq_len"])
    feature_cols = data["feature_cols"].tolist()
    target_col = str(data["target_col"])

    scalers = joblib.load(base_dir / scaler_file)
    feature_scaler: StandardScaler = scalers["feature_scaler"]
    target_scaler: StandardScaler = scalers["target_scaler"]

    print("Loaded preprocessed data:")
    print(f"  X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"  X_val:   {X_val.shape}, y_val:   {y_val.shape}")
    print(f"  X_test:  {X_test.shape}, y_test:  {y_test.shape}")
    print(f"  seq_len = {seq_len}, n_features = {X_train.shape[2]}")

    return (
        (X_train, y_train),
        (X_val, y_val),
        (X_test, y_test),
        feature_scaler,
        target_scaler,
        seq_len,
        feature_cols,
        target_col,
    )


class SequenceDataset(Dataset):
    """PyTorch Dataset for sequences."""

    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]


# ---------- Models ----------

class LSTMPredictor(nn.Module):
    """LSTM model for precipitation prediction."""

    def __init__(
            self,
            input_dim: int,
            hidden_dim: int = 64,
            num_layers: int = 2,
            dropout: float = 0.2,
    ):
        super().__init__()

        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Manual dropout for LSTM output (used when num_layers == 1)
        self.lstm_dropout = nn.Dropout(dropout)

        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, sequence_length, input_dim)
        lstm_out, _ = self.lstm(x)

        # Use last timestep
        last_output = lstm_out[:, -1, :]

        # Apply dropout to LSTM output if only 1 layer
        # (for multi-layer, dropout is already applied between LSTM layers)
        if self.num_layers == 1:
            last_output = self.lstm_dropout(last_output)

        # Predict
        output = self.fc(last_output)
        return output.squeeze(-1)


class GRUPredictor(nn.Module):
    """GRU model (lighter alternative to LSTM)."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 32,
        num_layers: int = 1,
        dropout: float = 0.4,
    ):
        super().__init__()

        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gru_out, _ = self.gru(x)
        last_output = gru_out[:, -1, :]
        output = self.fc(last_output)
        return output.squeeze(-1)


def train_model(
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 50,
        lr: float = 0.001,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
):
    """Train the model and return model + loss histories."""
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_loss = float('inf')
    best_state_dict = None

    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        # ----- Training -----
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            predictions = model(X_batch)
            loss = criterion(predictions, y_batch)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # ----- Validation -----
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                predictions = model(X_batch)
                loss = criterion(predictions, y_batch)
                val_loss += loss.item()

        val_loss /= len(val_loader)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(
            f"Epoch {epoch + 1}/{epochs} - "
            f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
        )

        # Save best model in memory
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state_dict = model.state_dict()

    # Load best model
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    return model, train_losses, val_losses

def evaluate_model(
    model: nn.Module,
    test_loader: DataLoader,
    target_scaler: StandardScaler,
    device: str | torch.device = None,
):
    """Evaluate model on test set, inverse-transforming targets."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

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
    predictions = target_scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()
    actuals = target_scaler.inverse_transform(actuals.reshape(-1, 1)).flatten()

    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

    rmse = mean_squared_error(actuals, predictions)
    mae = mean_absolute_error(actuals, predictions)
    r2 = r2_score(actuals, predictions)

    print("\nTest Results:")
    print(f"RMSE: {rmse:.3f} mm")
    print(f"MAE:  {mae:.3f} mm")
    print(f"R²:   {r2:.4f}")

    return predictions, actuals



if __name__ == "__main__":
    base_dir = "preprocessed"  # adjust if needed

    (X_train, y_train), (X_val, y_val), (X_test, y_test), _, target_scaler, seq_len, feature_cols, target_col = (
        load_preprocessed_artifacts(base_dir=base_dir)
    )

    train_dataset = SequenceDataset(X_train, y_train)
    val_dataset = SequenceDataset(X_val, y_val)
    test_dataset = SequenceDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    input_dim = X_train.shape[2]

    model = LSTMPredictor(
        input_dim=input_dim,
        hidden_dim=64,
        num_layers=2,
        dropout=0.2,
    )

    print(f"\nModel: LSTM with {input_dim} input features")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")

    model = train_model(
        model,
        train_loader,
        val_loader,
        epochs=50,
        lr=0.001,
    )

    predictions, actuals = evaluate_model(
        model,
        test_loader,
        target_scaler=target_scaler,
    )

"""Data loading, normalisation, and sliding window dataset. Supports ETTh1 and Weather."""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

# Standard ETTh1 split boundaries
TRAIN_END = 12 * 30 * 24       # 8640
VAL_END = TRAIN_END + 4 * 30 * 24  # 11520
TEST_END = VAL_END + 4 * 30 * 24   # 14400

FEATURE_COLS = ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"]


def load_etth1(data_path: str = "data/ETTh1.csv") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load ETTh1 and return train, val, test arrays of shape (n_rows, 7)."""
    df = pd.read_csv(data_path)
    data = df[FEATURE_COLS].values[:TEST_END].astype(np.float32)
    return data[:TRAIN_END], data[TRAIN_END:VAL_END], data[VAL_END:TEST_END]


def load_weather(data_path: str = "data/weather.csv") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load Weather dataset and return train, val, test arrays of shape (n_rows, 21).

    Split: 70% train, 10% val, 20% test (standard for Weather).
    """
    df = pd.read_csv(data_path)
    # Drop date column, keep all 21 numeric feature columns
    data = df.iloc[:, 1:].values.astype(np.float32)
    n = len(data)
    n_train = int(n * 0.7)
    n_test = int(n * 0.2)
    n_val = n - n_train - n_test
    return data[:n_train], data[n_train:n_train + n_val], data[n_train + n_val:]


def load_dataset(data_path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Auto-detect dataset from filename and load."""
    if "weather" in data_path.lower():
        return load_weather(data_path)
    else:
        return load_etth1(data_path)


class PerChannelScaler:
    """StandardScaler that fits per-channel (column) on train data."""

    def __init__(self):
        self.mean: np.ndarray | None = None
        self.std: np.ndarray | None = None

    def fit(self, data: np.ndarray) -> "PerChannelScaler":
        self.mean = data.mean(axis=0)
        self.std = data.std(axis=0)
        self.std[self.std == 0] = 1.0  # avoid division by zero
        return self

    def transform(self, data: np.ndarray) -> np.ndarray:
        return (data - self.mean) / self.std

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        return data * self.std + self.mean


class ETTh1Dataset(Dataset):
    """Sliding window dataset over contiguous normalised data.

    For val/test, pass the full normalised array and use border_start/border_end
    to define where forecast targets must fall. The lookback can reach before
    border_start (into the prior split).

    Returns:
        x: (n_channels, seq_len)  — multivariate input window
        y: (n_channels, pred_len) — multivariate target window
    """

    def __init__(self, data: np.ndarray, seq_len: int, pred_len: int,
                 border_start: int = 0, border_end: int | None = None):
        """
        Args:
            data: full normalised array of shape (total_rows, n_channels)
            seq_len: lookback length L
            pred_len: forecast horizon T
            border_start: first row index where forecast targets begin
            border_end: last row index (exclusive) for forecast targets
        """
        self.data = torch.from_numpy(data).float()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.border_start = border_start
        self.border_end = border_end if border_end is not None else len(data)

        # Number of valid windows: target must fit within [border_start, border_end)
        # Target for window i spans [border_start + i, border_start + i + pred_len)
        # Last valid i: border_start + i + pred_len <= border_end
        self.n_windows = self.border_end - self.border_start - pred_len + 1

    def __len__(self) -> int:
        return self.n_windows

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        # Target starts at border_start + idx
        target_start = self.border_start + idx
        input_start = target_start - self.seq_len

        x = self.data[input_start:target_start].T           # (channels, seq_len)
        y = self.data[target_start:target_start + self.pred_len].T  # (channels, pred_len)
        return x, y


def get_dataloaders(
    data_path: str = "data/ETTh1.csv",
    seq_len: int = 336,
    pred_len: int = 96,
    batch_size: int = 32,
) -> tuple[DataLoader, DataLoader, DataLoader, PerChannelScaler]:
    """Build train/val/test DataLoaders with proper normalisation."""
    train_raw, val_raw, test_raw = load_dataset(data_path)

    scaler = PerChannelScaler().fit(train_raw)
    all_data = np.concatenate([train_raw, val_raw, test_raw], axis=0)
    all_normalised = scaler.transform(all_data)

    # Compute split boundaries from actual data sizes
    train_end = len(train_raw)
    val_end = train_end + len(val_raw)

    train_ds = ETTh1Dataset(all_normalised, seq_len, pred_len,
                            border_start=seq_len, border_end=train_end)
    val_ds = ETTh1Dataset(all_normalised, seq_len, pred_len,
                          border_start=train_end, border_end=val_end)
    test_ds = ETTh1Dataset(all_normalised, seq_len, pred_len,
                           border_start=val_end, border_end=len(all_data))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, drop_last=False)

    n_channels = all_data.shape[1]
    return train_loader, val_loader, test_loader, scaler, n_channels


if __name__ == "__main__":
    # Quick verification
    for pred_len in [96, 192, 336, 720]:
        train_loader, val_loader, test_loader, scaler, n_ch = get_dataloaders(pred_len=pred_len)
        x, y = next(iter(train_loader))
        print(f"T={pred_len}: train={len(train_loader.dataset)}, "
              f"val={len(val_loader.dataset)}, test={len(test_loader.dataset)}, "
              f"x={tuple(x.shape)}, y={tuple(y.shape)}")

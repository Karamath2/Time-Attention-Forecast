# src/dataset.py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

class TimeSeriesWindowDataset(Dataset):
    def __init__(self, df, input_features, target_col, input_len=60, horizon=1, device='cpu'):
        """
        df: pandas DataFrame with time-ordered rows
        input_features: list of column names used as inputs
        target_col: column name for target
        input_len: number of time steps of history
        horizon: forecast horizon (1 means next step)
        """
        self.df = df.reset_index(drop=True)
        self.input_features = input_features
        self.target_col = target_col
        self.input_len = input_len
        self.horizon = horizon
        self.device = device

        self.X = self.df[input_features].values.astype(np.float32)
        self.y = self.df[target_col].values.astype(np.float32)

    def __len__(self):
        return len(self.df) - self.input_len - self.horizon + 1

    def __getitem__(self, idx):
        x = self.X[idx: idx + self.input_len]  # shape (input_len, n_features)
        y = self.y[idx + self.input_len + self.horizon - 1]  # scalar
        return torch.from_numpy(x), torch.tensor(y, dtype=torch.float32)

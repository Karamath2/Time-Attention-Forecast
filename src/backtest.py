# src/backtest.py
import pandas as pd
import numpy as np
from dataset import TimeSeriesWindowDataset
from models import LSTMWithAttention, LSTMPlain
from train import train_model
from torch.utils.data import DataLoader
from utils import evaluate_metrics
import torch
import os

def rolling_origin(df, input_features, target_col, initial_train_size, horizon=1, step=50, input_len=60, device="cpu"):
    """
    For each fold:
      - train on data[:train_end]
      - validate on next horizon (or a held-out val set)
      - collect predictions
    """
    n = len(df)
    results = []
    fold = 0
    for train_end in range(initial_train_size, n - horizon + 1, step):
        fold += 1
        train_df = df.iloc[:train_end]
        test_df = df.iloc[train_end: train_end + horizon]
        # Prepare datasets with small validation (last 10% of train)
        val_split = int(len(train_df) * 0.85)
        train_sub = train_df.iloc[:val_split]
        val_sub = train_df.iloc[val_split:]

        train_ds = TimeSeriesWindowDataset(train_sub, input_features, target_col, input_len=input_len)
        val_ds = TimeSeriesWindowDataset(val_sub, input_features, target_col, input_len=input_len)
        test_ds = TimeSeriesWindowDataset(pd.concat([train_df.iloc[-input_len:], test_df]), input_features, target_col, input_len=input_len)

        train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)
        test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)

        model = LSTMWithAttention(n_features=len(input_features), hidden_dim=64).to(device)
        os.makedirs("models", exist_ok=True)
        model_path = f"models/attn_fold_{fold}.pt"
        _ = train_model(model, train_loader, val_loader, n_epochs=10, lr=1e-3, device=device, model_path=model_path)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        y_true, y_pred = [], []
        with torch.no_grad():
            for X, y in test_loader:
                X = X.to(device)
                outputs = model(X)
                if isinstance(outputs, tuple):
                    pred = outputs[0]
                else:
                    pred = outputs
                y_true.append(y.item())
                y_pred.append(pred.item())
        metrics = evaluate_metrics(y_true, y_pred)
        print(f"Fold {fold} train_end={train_end} metrics={metrics}")
        results.append(metrics)
    # Aggregate
    return results

if __name__ == "__main__":
    df = pd.read_csv("data/generated.csv")
    input_features = ["feature1","feature2","feature3","feature4","exog"]
    results = rolling_origin(df, input_features, "target", initial_train_size=500, horizon=10, step=200, input_len=60)
    print("Aggregated results:", results)

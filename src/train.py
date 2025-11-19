# src/train.py
"""
Train script for LSTMWithAttention or LSTMPlain.
- Expects data/generated.csv to exist (created by data_gen.py)
- Saves model checkpoint to models/attn_model.pt (or plain_model.pt)
- Saves scalers to models/input_scaler.joblib and models/target_scaler.joblib
"""
import os
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import joblib

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import TimeSeriesWindowDataset
from models import LSTMWithAttention, LSTMPlain
from utils import evaluate_metrics

from sklearn.preprocessing import StandardScaler
from typing import Tuple, Dict, Any

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

def create_dirs(path="models"):
    os.makedirs(path, exist_ok=True)

def scale_datasets(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame,
                   input_features, target_col, models_dir="models") -> Tuple[pd.DataFrame,pd.DataFrame,pd.DataFrame]:
    """
    Fit scalers on train_df and transform val/test. Save scalers to models_dir.
    Returns scaled copies of dataframes (non-destructive).
    """
    create_dirs(models_dir)
    input_scaler = StandardScaler()
    target_scaler = StandardScaler()

    # Fit on training inputs and target
    X_train = train_df[input_features].values.astype(float)
    y_train = train_df[[target_col]].values.astype(float)

    input_scaler.fit(X_train)
    target_scaler.fit(y_train)

    # transform and produce new dfs
    train_scaled = train_df.copy()
    val_scaled = val_df.copy()
    test_scaled = test_df.copy()

    train_scaled[input_features] = input_scaler.transform(train_df[input_features].values)
    val_scaled[input_features] = input_scaler.transform(val_df[input_features].values)
    test_scaled[input_features] = input_scaler.transform(test_df[input_features].values)

    train_scaled[[target_col]] = target_scaler.transform(train_df[[target_col]].values)
    val_scaled[[target_col]] = target_scaler.transform(val_df[[target_col]].values)
    test_scaled[[target_col]] = target_scaler.transform(test_df[[target_col]].values)

    # Save scalers
    joblib.dump(input_scaler, os.path.join(models_dir, "input_scaler.joblib"))
    joblib.dump(target_scaler, os.path.join(models_dir, "target_scaler.joblib"))
    print(f"Saved scalers to {models_dir}")

    return train_scaled, val_scaled, test_scaled

def get_dataloaders(train_df, val_df, test_df, input_features, target_col,
                    input_len=60, batch_size=64, num_workers=0):
    train_ds = TimeSeriesWindowDataset(train_df, input_features, target_col, input_len=input_len)
    val_ds = TimeSeriesWindowDataset(val_df, input_features, target_col, input_len=input_len)
    test_ds = TimeSeriesWindowDataset(test_df, input_features, target_col, input_len=input_len)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader

def train_model(model: nn.Module,
                train_loader: DataLoader,
                val_loader: DataLoader,
                n_epochs: int = 30,
                lr: float = 1e-3,
                device: str = "cpu",
                model_path: str = "models/attn_model.pt") -> nn.Module:
    """
    Train model and save best checkpoint by validation loss.
    """
    create_dirs(os.path.dirname(model_path) or "models")
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    best_val_loss = float("inf")

    for epoch in range(1, n_epochs + 1):
        model.train()
        train_losses = []
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{n_epochs}", leave=False)
        for X, y in pbar:
            X = X.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            outputs = model(X)
            # model with attention returns (pred, attn) while plain returns pred
            if isinstance(outputs, tuple):
                pred = outputs[0]
            else:
                pred = outputs
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
            pbar.set_postfix(train_loss=f"{np.mean(train_losses):.6f}")

        # validation
        model.eval()
        val_losses = []
        y_true, y_pred = [], []
        with torch.no_grad():
            for X, y in val_loader:
                X = X.to(device)
                y = y.to(device)
                outputs = model(X)
                if isinstance(outputs, tuple):
                    pred = outputs[0]
                else:
                    pred = outputs
                val_losses.append(criterion(pred, y).item())
                y_true.extend(y.cpu().numpy().tolist())
                y_pred.extend(pred.cpu().numpy().tolist())

        val_loss = float(np.mean(val_losses)) if val_losses else float("nan")
        metrics = evaluate_metrics(np.array(y_true).ravel(), np.array(y_pred).ravel()) if y_true else {}
        print(f"Epoch {epoch}  train_loss={np.mean(train_losses):.6f}  val_loss={val_loss:.6f}  RMSE={metrics.get('RMSE', np.nan):.4f}  MAE={metrics.get('MAE', np.nan):.4f}  MAPE={metrics.get('MAPE', np.nan):.2f}%")

        # save if improved
        if not np.isnan(val_loss) and val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_path)
            # also save optimizer state if desired (optional)
    # load best model before returning if exists
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
    return model

def evaluate_on_test(model: nn.Module, test_loader: DataLoader, device="cpu", target_scaler_path=None) -> Dict[str, Any]:
    """
    Evaluates model on test_loader and returns metrics and raw arrays.
    If target_scaler_path provided, inverse transforms predictions & truths.
    """
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

    y_true = np.array(y_true).reshape(-1, 1)
    y_pred = np.array(y_pred).reshape(-1, 1)

    # inverse transform if scaler provided
    if target_scaler_path and os.path.exists(target_scaler_path):
        scaler = joblib.load(target_scaler_path)
        y_true = scaler.inverse_transform(y_true).ravel()
        y_pred = scaler.inverse_transform(y_pred).ravel()
    else:
        y_true = y_true.ravel()
        y_pred = y_pred.ravel()

    metrics = evaluate_metrics(y_true, y_pred)
    return {"metrics": metrics, "y_true": y_true, "y_pred": y_pred}

def main(args):
    # Load data
    df = pd.read_csv(args.data_path)
    input_features = args.input_features.split(",")
    target_col = args.target_col

    # train/val/test splits (time-based)
    n = len(df)
    train_end = int(n * 0.7)
    val_end = int(n * 0.85)

    train_df = df.iloc[:train_end].reset_index(drop=True)
    val_df = df.iloc[train_end:val_end].reset_index(drop=True)
    test_df = df.iloc[val_end:].reset_index(drop=True)

    # scale datasets and save scalers
    train_s, val_s, test_s = scale_datasets(train_df, val_df, test_df, input_features, target_col, models_dir=args.models_dir)

    # dataloaders
    train_loader, val_loader, test_loader = get_dataloaders(train_s, val_s, test_s, input_features, target_col,
                                                           input_len=args.input_len, batch_size=args.batch_size, num_workers=0)

    # model selection
    n_features = len(input_features)
    if args.model_type == "attn":
        model = LSTMWithAttention(n_features=n_features, hidden_dim=args.hidden_dim, n_layers=args.n_layers, attn_dim=args.attn_dim, dropout=args.dropout)
        model_path = os.path.join(args.models_dir, args.model_filename or "attn_model.pt")
    else:
        model = LSTMPlain(n_features=n_features, hidden_dim=args.hidden_dim, n_layers=args.n_layers, dropout=args.dropout)
        model_path = os.path.join(args.models_dir, args.model_filename or "plain_model.pt")

    print(f"Training model: {args.model_type} on device {args.device}")
    create_dirs(args.models_dir)

    trained = train_model(model, train_loader, val_loader, n_epochs=args.epochs, lr=args.lr, device=args.device, model_path=model_path)

    # evaluate on test
    target_scaler_path = os.path.join(args.models_dir, "target_scaler.joblib")
    result = evaluate_on_test(trained, test_loader, device=args.device, target_scaler_path=target_scaler_path)
    print("Test metrics:", result["metrics"])

    # save final predictions (inverse scaled) to CSV for inspection
    out_df = pd.DataFrame({"y_true": result["y_true"], "y_pred": result["y_pred"]})
    out_csv = os.path.join(args.models_dir, "test_predictions.csv")
    out_df.to_csv(out_csv, index=False)
    print(f"Saved test predictions to {out_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="data/generated.csv", help="Path to generated.csv")
    parser.add_argument("--input_features", type=str, default="feature1,feature2,feature3,feature4,exog", help="Comma-separated input features")
    parser.add_argument("--target_col", type=str, default="target", help="Target column")
    parser.add_argument("--input_len", type=int, default=60, help="History window length")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--attn_dim", type=int, default=32)
    parser.add_argument("--n_layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--model_type", type=str, choices=["attn", "plain"], default="attn")
    parser.add_argument("--models_dir", type=str, default="models")
    parser.add_argument("--model_filename", type=str, default=None)
    parser.add_argument("--device", type=str, default="cpu", help="cpu or cuda")
    args = parser.parse_args()
    # auto detect gpu if requested
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but not available. Using CPU.")
        args.device = "cpu"

    main(args)

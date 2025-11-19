# src/utils.py
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def mape(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0

def evaluate_metrics(y_true, y_pred):
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
        "MAPE": mape(y_true, y_pred)
    }

# Simple plotting helper
import matplotlib.pyplot as plt
def plot_attention_weights(attn, title="Attention weights"):
    plt.figure(figsize=(8,3))
    plt.imshow(attn[np.newaxis, :], aspect="auto")
    plt.yticks([])
    plt.xlabel("Time step (history)")
    plt.title(title)
    plt.colorbar()
    plt.show()

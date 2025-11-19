import torch
import joblib
import pandas as pd
import os
import matplotlib.pyplot as plt

from src.models import LSTMWithAttention
from src.dataset import TimeSeriesWindowDataset
from src.utils import plot_attention_weights

# create folder
os.makedirs("outputs/plots", exist_ok=True)

# load scalers
input_scaler = joblib.load("models/input_scaler.joblib")
target_scaler = joblib.load("models/target_scaler.joblib")

# load model
model = LSTMWithAttention(n_features=5, hidden_dim=128)
model.load_state_dict(torch.load("models/attn_model.pt", map_location="cpu"))
model.eval()

# prepare window
df = pd.read_csv("data/generated.csv")
input_features = ["feature1","feature2","feature3","feature4","exog"]
input_len = 60

# FIX: Take last 61 rows
window = df.iloc[-(input_len+1):].copy()

# scale
window[input_features] = input_scaler.transform(window[input_features])

# dataset window
ds = TimeSeriesWindowDataset(window.reset_index(drop=True), input_features, "target", input_len=input_len)
X, y = ds[0]

# run model
with torch.no_grad():
    pred, attn = model(X.unsqueeze(0))
    pred_inv = target_scaler.inverse_transform([[pred.item()]])[0][0]

print("Prediction:", pred_inv)

# plot and save attention weights manually
plt.figure(figsize=(10, 3))
plt.imshow(attn.squeeze().cpu().numpy()[None, :], aspect="auto", cmap="viridis")
plt.title("Attention Weights (Last Window)")
plt.xlabel("Time Step")
plt.yticks([])

save_path = "outputs/plots/attention_weights.png"
plt.savefig(save_path, dpi=300)
print(f"Saved attention plot to {save_path}")

plt.show()

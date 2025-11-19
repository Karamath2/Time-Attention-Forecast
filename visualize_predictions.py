import pandas as pd
import matplotlib.pyplot as plt
import os

# create folder if not exists
os.makedirs("outputs/plots", exist_ok=True)

# load predictions
df = pd.read_csv("models/test_predictions.csv")

# plot
plt.figure(figsize=(10,4))
plt.plot(df['y_true'], label='True')
plt.plot(df['y_pred'], label='Predicted')
plt.legend()
plt.title("Test Predictions: True vs Predicted")
plt.xlabel("Time")
plt.ylabel("Value")
plt.grid(True)

# save figure
save_path = "outputs/plots/predictions_plot.png"
plt.savefig(save_path, dpi=300)
print(f"Saved prediction plot to {save_path}")

plt.show()

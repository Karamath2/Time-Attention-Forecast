# src/data_gen.py
import numpy as np
import pandas as pd

def generate_multivariate_ts(n_steps=1500, seed=42):
    np.random.seed(seed)
    t = np.arange(n_steps)
    # Feature 1: seasonal (yearly-like) + trend
    f1 = 0.05 * t + 2.0 * np.sin(2 * np.pi * t / 50) + 0.5 * np.random.randn(n_steps)
    # Feature 2: seasonal with different period + noise
    f2 = -0.03 * t + 1.5 * np.sin(2 * np.pi * t / 20 + 0.5) + 0.3 * np.random.randn(n_steps)
    # Feature 3: correlated with f1 (lagged)
    f3 = 0.8 * np.roll(f1, 3) + 0.2 * np.random.randn(n_steps)
    # Feature 4: exogenous event (spike)
    exog = np.zeros(n_steps)
    for center in [400, 900, 1200]:
        exog += 10.0 * np.exp(-0.5 * ((t - center) / 10) ** 2)
    f4 = 0.02 * t + exog + 0.2 * np.random.randn(n_steps)
    # Target: a combination (we forecast target)
    target = 1.2 * f1 - 0.6 * f2 + 0.4 * f3 + 0.7 * f4 + 0.3 * np.random.randn(n_steps)

    df = pd.DataFrame({
        "t": t,
        "feature1": f1,
        "feature2": f2,
        "feature3": f3,
        "feature4": f4,
        "exog": exog,
        "target": target
    })
    return df

if __name__ == "__main__":
    df = generate_multivariate_ts()
    df.to_csv("data/generated.csv", index=False)
    print("Saved data/generated.csv with shape", df.shape)

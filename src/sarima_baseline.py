# src/sarima_baseline.py
import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from utils import evaluate_metrics

def sarima_fit_forecast(train_series, test_series, exog_train=None, exog_test=None, order=(1,0,1), seasonal_order=(0,0,0,0)):
    model = SARIMAX(train_series, exog=exog_train, order=order, seasonal_order=seasonal_order, enforce_stationarity=False, enforce_invertibility=False)
    res = model.fit(disp=False)
    preds = res.get_forecast(steps=len(test_series), exog=exog_test).predicted_mean
    return preds

if __name__ == "__main__":
    df = pd.read_csv("data/generated.csv")
    n = len(df)
    train_end = int(n*0.7)
    val_end = int(n*0.85)
    train = df.iloc[:train_end]
    test = df.iloc[val_end:]
    preds = sarima_fit_forecast(train["target"], test["target"], exog_train=train[["exog"]], exog_test=test[["exog"]], order=(1,1,1))
    metrics = evaluate_metrics(test["target"].values, preds.values)
    print(metrics)

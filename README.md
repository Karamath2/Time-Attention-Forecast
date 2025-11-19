📘 README.md (Complete Version)
Time Series Forecasting with LSTM + Attention, SARIMA, and Backtesting
🧠 Time-Attention-Forecast

A complete machine learning project for multivariate time-series forecasting using:

LSTM

LSTM with Bahdanau Attention

SARIMA baseline

Rolling-Origin Backtesting

Programmatically generated dataset

Prediction & Attention visualizations

This project builds, trains, evaluates, and compares multiple forecasting models on synthetic time-series data generated using seasonal, trend, noise, and exogenous patterns.

📂 Project Structure
time-attention-forecast/
├── data/
│   └── generated.csv
├── models/
│   ├── attn_model.pt
│   ├── input_scaler.joblib
│   ├── target_scaler.joblib
│   └── test_predictions.csv
├── notebooks/
│   └── EDA_and_plots.ipynb (optional)
├── src/
│   ├── data_gen.py
│   ├── dataset.py
│   ├── models.py
│   ├── train.py
│   ├── backtest.py
│   ├── sarima_baseline.py
│   └── utils.py
├── visualize_predictions.py
├── visualize_attention.py
├── requirements.txt
└── README.md

📊 1. Dataset Generation

The project uses a programmatically generated multivariate time-series dataset created with:

Seasonal components

Trends

Exogenous features

Noise

Spikes/events

Run:

python src/data_gen.py


It generates:

data/generated.csv

🔧 2. Models Implemented
✔️ LSTM Baseline

Standard LSTM model for sequence forecasting.

✔️ LSTM with Bahdanau Attention

Enhances the LSTM by learning:

Which timesteps to focus on

How much influence past steps have

Helps interpret temporal importance.

✔️ SARIMA Baseline

A statistical baseline for comparison.

✔️ Rolling-Origin Cross Validation

Multiple expanding-window folds to evaluate forecasting stability.

🏋️‍♂️ 3. Training the Attention Model

Run:

python src/train.py


This script:

Loads & splits data

Standardizes inputs/target

Trains LSTM with Attention

Saves model + scalers

Evaluates on test set

Saves predictions to:

models/test_predictions.csv

📈 4. Evaluation Results
LSTM with Attention — Final Test Metrics

(Your results may vary slightly)

Metric	Value
MAE	~25.47
RMSE	~25.89
MAPE	~16.57%
SARIMA Baseline
Metric	Value
MAE	~36.57
RMSE	~37.33
MAPE	~24.25%

→ LSTM with Attention outperforms SARIMA on this dataset.

🔄 5. Backtesting (Rolling Origin Evaluation)

Run:

python src/backtest.py


This evaluates model performance across multiple folds with increasing training windows.

📉 6. Visualization Tools
📌 A) Prediction Plot

Run:

python visualize_predictions.py


Shows:

True values

Model predictions

📌 B) Attention Heatmap

Run:

python visualize_attention.py


Shows:

Attention weights across 60 historical timesteps

Which parts of history the model focused on

Great for model interpretability.

🛠️ 7. Installation

Install dependencies:

pip install -r requirements.txt


Run inside virtual environment recommended.

🚀 8. How to Run Entire Pipeline
python src/data_gen.py
python src/train.py
python src/sarima_baseline.py
python src/backtest.py
python visualize_predictions.py
python visualize_attention.py

📜 9. Key Features

✔️ End-to-end dataset generation
✔️ Deep learning + statistical models
✔️ Attention visualization
✔️ Backtesting for robust validation
✔️ Clean modular project structure
✔️ Ready for research or deployment

📎 10. Requirements

See requirements.txt

🤝 Contributing

Pull requests are welcome. Open an issue to discuss new ideas.

📄 License

This project is open-source under the MIT License.
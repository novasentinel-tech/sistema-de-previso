# 🔮 TOTEM_DEEPSEA - Multivariate Time Series Forecasting System

A complete, local-first system for multivariate time series forecasting using **LSTM** neural networks and **Facebook Prophet**, with interactive dashboards and comprehensive evaluation tools.

## ✨ Features

- **🤖 LSTM Neural Networks**: Deep learning models for complex temporal patterns
- **🔮 Facebook Prophet**: Univariate forecasting with seasonality handling
- **📊 Interactive Dashboard**: Streamlit interface for visualization and prediction
- **📈 Multiple Metrics**: MAE, RMSE, MAPE, R² for model evaluation
- **🎯 Feature Engineering**: Automatic temporal and statistical feature creation
- **🧹 Data Preprocessing**: Cleaning, normalization, and sequence creation
- **📁 100% Local**: No Firebase or cloud dependencies—everything runs locally
- **🧪 Unit Tests**: Comprehensive test suite with pytest

---

## 📁 Project Structure

```
TOTEM_DEEPSEA/
│
├── data/
│   ├── raw/                          # Place your CSV files here
│   └── processed/                    # Preprocessed data (auto-generated)
│
├── notebooks/                        # Jupyter notebooks for exploration
│
├── src/                             # Main source code
│   ├── __init__.py
│   ├── config.py                    # Global configuration
│   ├── data_preprocessing.py        # Data cleaning and feature engineering
│   ├── evaluation.py                # Model evaluation and metrics
│   ├── prediction.py                # Inference engine
│   └── models/
│       ├── __init__.py
│       ├── lstm_model.py            # LSTM model definition
│       ├── prophet_model.py         # Prophet model wrapper
│       ├── train.py                 # Training pipeline
│       └── saved/                   # Trained models (auto-generated)
│
├── dashboard/                       # Streamlit dashboard
│   ├── streamlit_app.py            # Main dashboard app
│   └── plotly_charts.py            # Interactive visualizations
│
├── tests/                          # Unit tests
│   ├── test_preprocessing.py
│   ├── test_models.py
│   └── test_prediction.py
│
├── requirements.txt                # Python dependencies
├── README.md                       # This file
└── .gitignore                      # Git ignore rules
```

---

## 🚀 Getting Started

### 1. Installation

```bash
# Clone or download the project
cd TOTEM_DEEPSEA

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Prepare Your Data

Add your CSV files to the `data/raw/` folder. Expected format:
- CSV with datetime index or datetime column
- Numeric columns for time series values
- Example: `energy_consumption.csv`, `stock_prices.csv`, `traffic_data.csv`

### 3. Train Models

```bash
# Option A: Run full pipeline (recommended)
python -m src.models.train

# Option B: Use Jupyter notebooks for exploration
jupyter notebook
# Open notebooks/train_model.ipynb
```

### 4. View Results

```bash
# Launch interactive dashboard
streamlit run dashboard/streamlit_app.py
```

The dashboard will open at `http://localhost:8501`

---

## 📖 Module Documentation

### `config.py`
Global configuration file with all hyperparameters:
- **LSTM Settings**: units, dropout, learning rate, epochs
- **Prophet Settings**: seasonality, interval width
- **Data Parameters**: test/validation split ratios
- **Paths**: data directories, model paths

**Key Variables:**
```python
LSTM_LOOKBACK = 24              # 24 timesteps for LSTM
LSTM_EPOCHS = 100
FORECAST_HORIZON = 24            # Predict next 24 steps
NORMALIZATION_METHOD = 'minmax'
```

### `data_preprocessing.py`
Handles all data operations:

```python
from src.data_preprocessing import DataPreprocessor

preprocessor = DataPreprocessor()

# Load raw data
df = preprocessor.load_raw_data('my_data.csv')

# Clean and feature engineer
df = preprocessor.clean_data(df)
df = preprocessor.create_features(df)

# Prepare for modeling
numeric_data = df.select_dtypes(include=[np.number]).values
X_train_scaled, X_test_scaled, _ = preprocessor.normalize_data(X_train, X_test)
X, y = preprocessor.create_sequences(X_train_scaled, lookback=24)
X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.train_test_split(X, y)
```

**Main Functions:**
- `load_raw_data(filename)` - Load CSV
- `clean_data(df)` - Remove NaNs and outliers
- `create_features(df, datetime_col)` - Generate temporal features
- `normalize_data(X_train, X_test)` - MinMax or Standard scaling
- `create_sequences(data, lookback)` - Create LSTM sequences
- `train_test_split(X, y)` - Split into train/val/test

### `models/lstm_model.py`
LSTM neural network implementation:

```python
from src.models.lstm_model import build_lstm_model, train_lstm

# Build model
model = build_lstm_model(input_shape=(24, 5))  # 24 timesteps, 5 features

# Train model
model, history = train_lstm(X_train, y_train, X_val, y_val, model_name='my_model')

# Make predictions
predictions = model.predict(X_test)
```

**Architecture:**
- 2 LSTM layers with dropout
- Dense layers for output
- Adam optimizer with MSE loss

### `models/prophet_model.py`
Facebook Prophet for univariate forecasting:

```python
from src.models.prophet_model import train_prophet, forecast_prophet

# Train for single column
model = train_prophet(df, col_target='energy_consumption', model_name='energy')

# Forecast
forecast = forecast_prophet(model, periods=24)
# Returns: DataFrame with 'yhat', 'yhat_lower', 'yhat_upper'

# Train for all columns
models_dict = train_prophet_for_all_columns(df, prefix='my_models')
```

### `evaluation.py`
Model evaluation and visualization:

```python
from src.evaluation import calculate_metrics, plot_predictions, create_evaluation_report

# Calculate metrics
metrics = calculate_metrics(y_true, y_pred)
# Returns: {'mae': ..., 'rmse': ..., 'mape': ..., 'r2': ...}

# Create plots
plot_predictions(y_true, y_pred, title='LSTM Predictions')
plot_error_distribution(y_true, y_pred)
plot_residuals(y_true, y_pred)

# Generate report
report = create_evaluation_report(y_true, y_pred, model_name='LSTM')
```

### `prediction.py`
Inference engine for making predictions:

```python
from src.prediction import PredictionEngine

# Initialize engine with trained models
engine = PredictionEngine('my_model_prefix')

# LSTM predictions
predictions = engine.predict_lstm(X_test, inverse_scale=True)

# Prophet predictions
forecast = engine.predict_prophet('energy_consumption', periods=24)

# All models
results = engine.predict_all('new_data.csv')
```

### `dashboard/plotly_charts.py`
Interactive Plotly visualizations:

```python
from dashboard.plotly_charts import (
    create_time_series_plot,
    create_forecast_plot,
    create_prediction_vs_actual,
    create_metrics_comparison
)

# Time series
fig = create_time_series_plot(df, ['col1', 'col2'])
fig.show()

# Forecast vs history
fig = create_forecast_plot(df_historical, df_forecast, 'energy')
fig.show()

# Metrics
fig = create_metrics_comparison({'MAE': 0.05, 'RMSE': 0.08})
fig.show()
```

---

## 🤖 Complete Workflow Example

```python
# Step 1: Preprocess data
from src.data_preprocessing import quick_preprocess

X_train, X_val, X_test, y_train, y_val, y_test, scaler = \
    quick_preprocess('energy_data.csv', lookback=24)

# Step 2: Train LSTM
from src.models.lstm_model import train_lstm

model, history = train_lstm(X_train, y_train, X_val, y_val, 'energy_lstm')

# Step 3: Evaluate
from src.evaluation import calculate_metrics, plot_predictions

y_pred = model.predict(X_test)
metrics = calculate_metrics(y_test, y_pred)

print(f"MAE: {metrics['mae']:.4f}")
print(f"RMSE: {metrics['rmse']:.4f}")

plot_predictions(y_test, y_pred)

# Step 4: Make future predictions
from src.prediction import PredictionEngine

engine = PredictionEngine('energy_lstm')
future_predictions = engine.predict_lstm(X_new_data)

# Step 5: Visualize
from dashboard.plotly_charts import create_prediction_vs_actual

fig = create_prediction_vs_actual(y_test[:100], y_pred[:100])
fig.show()
```

---

## 🧪 Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_preprocessing.py -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

---

## 📊 Dashboard Features

The Streamlit dashboard includes:

1. **📊 Data Exploration**
   - Load and preview CSV files
   - Statistical summary
   - Interactive time series visualization

2. **🤖 Model Training**
   - Train LSTM and Prophet models
   - Monitor training metrics
   - Save trained models

3. **🔮 Predictions**
   - Load trained models
   - Generate forecasts
   - Compare different models

4. **📈 Evaluation**
   - Calculate performance metrics
   - View prediction vs actual plots
   - Analyze residuals and errors

---

## ⚙️ Configuration

Edit `src/config.py` to customize:

```python
# LSTM Parameters
LSTM_UNITS = [64, 32]           # Layer sizes
LSTM_DROPOUT = 0.2
LSTM_EPOCHS = 100
LSTM_LEARNING_RATE = 0.001
LSTM_LOOKBACK = 24              # Sequence length

# Prophet Parameters
PROPHET_YEARLY_SEASONALITY = True
PROPHET_INTERVAL_WIDTH = 0.95   # 95% confidence

# Data Parameters
TEST_SIZE = 0.2
NORMALIZATION_METHOD = 'minmax'
REMOVE_OUTLIERS = True
OUTLIER_THRESHOLD = 3
```

---

## 📝 Data Format

**Expected CSV format:**

| datetime | feature1 | feature2 | feature3 |
|----------|----------|----------|----------|
| 2024-01-01 00:00 | 100.5 | 45.2 | 1013.2 |
| 2024-01-01 01:00 | 101.2 | 44.8 | 1013.5 |
| 2024-01-01 02:00 | 99.8 | 46.1 | 1012.9 |

**Or with index:**
```
,temperature,humidity,pressure
2024-01-01 00:00,100.5,45.2,1013.2
2024-01-01 01:00,101.2,44.8,1013.5
```

---

## 🎯 Common Use Cases

### Energy Consumption Forecasting
```python
X, y = preprocess('energy.csv', 'timestamp')
model, _ = train_lstm(X, y)
forecast = predict_lstm(model, X_new)
```

### Stock Price Prediction
```python
X, y = preprocess('stocks.csv', lookback=20)
model, _ = train_lstm(X, y)
```

### Traffic Flow Prediction
```python
X, y = preprocess('traffic.csv')
model, _ = train_lstm(X, y, epochs=200)
```

---

## 🔧 Troubleshooting

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError: No module named 'tensorflow'` | Run `pip install -r requirements.txt` |
| `FileNotFoundError: Raw data not found` | Place CSV in `data/raw/` folder |
| `Model file not found` | Train model first with `train_lstm()` |
| Dashboard won't load | Run `pip install streamlit` and `streamlit run dashboard/streamlit_app.py` |
| Out of memory | Reduce `LSTM_BATCH_SIZE` in config.py |

---

## 📚 Additional Resources

- **TensorFlow/Keras**: https://keras.io/
- **Prophet Documentation**: https://facebook.github.io/prophet/
- **Streamlit**: https://streamlit.io/
- **Plotly**: https://plotly.com/

---

## 📄 License

This project is open source and available under the MIT License.

---

## 🤝 Contributing

Contributions are welcome! To contribute:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

---

## 📞 Support

For issues, questions, or suggestions, please create an issue in the repository.

---

**Built with 💚 by Jota**

*TOTEM_DEEPSEA v1.0 - 2024*

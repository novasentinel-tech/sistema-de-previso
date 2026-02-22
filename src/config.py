"""
Global Configuration for TOTEM_DEEPSEA
All paths, parameters, and hyperparameters are defined here.
"""

import os
from datetime import datetime

# ============================================================
# PATHS
# ============================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, 'data')
RAW_PATH = os.path.join(DATA_DIR, 'raw')
PROCESSED_PATH = os.path.join(DATA_DIR, 'processed')
MODELS_PATH = os.path.join(BASE_DIR, 'src', 'models', 'saved')
RESULTS_PATH = os.path.join(BASE_DIR, 'results')
LOGS_PATH = os.path.join(BASE_DIR, 'logs')

# Create directories if they don't exist
for path in [RAW_PATH, PROCESSED_PATH, MODELS_PATH, RESULTS_PATH, LOGS_PATH]:
    os.makedirs(path, exist_ok=True)

# ============================================================
# RANDOM SEED (Reproducibility)
# ============================================================
RANDOM_SEED = 42

# ============================================================
# DATA PREPROCESSING PARAMETERS
# ============================================================
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.1
NORMALIZATION_METHOD = 'minmax'  # 'minmax' or 'standard'
HANDLE_MISSING = 'forward_fill'  # 'drop', 'forward_fill', 'interpolate'
REMOVE_OUTLIERS = True
OUTLIER_THRESHOLD = 3  # Standard deviations

# ============================================================
# LSTM MODEL PARAMETERS
# ============================================================
LSTM_UNITS = [64, 32]  # Number of units in each LSTM layer
LSTM_DROPOUT = 0.2
LSTM_DENSE_UNITS = 16
LSTM_LOOKBACK = 24  # 24 timesteps (e.g., 24 hours)
LSTM_BATCH_SIZE = 16
LSTM_EPOCHS = 100
LSTM_LEARNING_RATE = 0.0005
LSTM_EARLY_STOPPING = True
LSTM_PATIENCE = 15

# ============================================================
# PROPHET MODEL PARAMETERS
# ============================================================
PROPHET_YEARLY_SEASONALITY = True
PROPHET_WEEKLY_SEASONALITY = True
PROPHET_DAILY_SEASONALITY = False
PROPHET_CHANGEPOINT_PRIOR_SCALE = 0.05
PROPHET_SEASONALITY_SCALE = 10
PROPHET_INTERVAL_WIDTH = 0.95  # 95% confidence interval

# ============================================================
# EVALUATION METRICS
# ============================================================
METRICS = ['mae', 'rmse', 'mape', 'r2']

# ============================================================
# PREDICTION SETTINGS
# ============================================================
FORECAST_HORIZON = 24  # Predict next 24 timesteps
CONFIDENCE_LEVEL = 0.95

# ============================================================
# DASHBOARD SETTINGS
# ============================================================
STREAMLIT_THEME = 'dark'
PLOT_HEIGHT = 500
PLOT_WIDTH = 1000

# ============================================================
# LOGGING
# ============================================================
LOG_LEVEL = 'INFO'
LOG_FORMAT = '[%(asctime)s - %(name)s - %(levelname)s]: %(message)s'

print(f"✅ Configuration loaded from: {os.path.abspath(__file__)}")
print(f"📁 Base directory: {BASE_DIR}")
print(f"🌱 Random seed: {RANDOM_SEED}")

import os
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, 'data')
RAW_PATH = os.path.join(DATA_DIR, 'raw')
PROCESSED_PATH = os.path.join(DATA_DIR, 'processed')
MODELS_PATH = os.path.join(BASE_DIR, 'src', 'models', 'saved')
RESULTS_PATH = os.path.join(BASE_DIR, 'results')
LOGS_PATH = os.path.join(BASE_DIR, 'logs')

for path in [RAW_PATH, PROCESSED_PATH, MODELS_PATH, RESULTS_PATH, LOGS_PATH]:
    os.makedirs(path, exist_ok=True)

RANDOM_SEED = 42

TEST_SIZE = 0.2
VALIDATION_SIZE = 0.1
NORMALIZATION_METHOD = 'minmax'
HANDLE_MISSING = 'forward_fill'
REMOVE_OUTLIERS = True
OUTLIER_THRESHOLD = 3

LSTM_UNITS = [64, 32]
LSTM_DROPOUT = 0.2
LSTM_DENSE_UNITS = 16
LSTM_LOOKBACK = 24
LSTM_BATCH_SIZE = 16
LSTM_EPOCHS = 100
LSTM_LEARNING_RATE = 0.0005
LSTM_EARLY_STOPPING = True
LSTM_PATIENCE = 15

PROPHET_YEARLY_SEASONALITY = True
PROPHET_WEEKLY_SEASONALITY = True
PROPHET_DAILY_SEASONALITY = False
PROPHET_CHANGEPOINT_PRIOR_SCALE = 0.05
PROPHET_SEASONALITY_SCALE = 10
PROPHET_INTERVAL_WIDTH = 0.95

METRICS = ['mae', 'rmse', 'mape', 'r2']

FORECAST_HORIZON = 24
CONFIDENCE_LEVEL = 0.95

STREAMLIT_THEME = 'dark'
PLOT_HEIGHT = 500
PLOT_WIDTH = 1000

LOG_LEVEL = 'INFO'
LOG_FORMAT = '[%(asctime)s - %(name)s - %(levelname)s]: %(message)s'

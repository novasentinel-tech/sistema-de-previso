"""
Prophet Model for Time Series Forecasting
Facebook's Prophet model, useful for univariate series with clear seasonality
"""

import pandas as pd
import numpy as np
import os
from prophet import Prophet
from config import (
    PROPHET_YEARLY_SEASONALITY, PROPHET_WEEKLY_SEASONALITY,
    PROPHET_DAILY_SEASONALITY, PROPHET_CHANGEPOINT_PRIOR_SCALE,
    PROPHET_SEASONALITY_SCALE, PROPHET_INTERVAL_WIDTH, MODELS_PATH, RANDOM_SEED
)
import logging
import pickle

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def train_prophet(df, col_target, col_datetime='ds', model_name=None):
    """
    Train Prophet model for a specific time series column
    
    Args:
        df (pd.DataFrame): Input dataframe with datetime index
        col_target (str): Column name containing values to forecast
        col_datetime (str): Column name containing datetime (default 'ds')
        model_name (str): Name for saving the model
        
    Returns:
        Prophet.model: Trained Prophet model
        
    Example:
        >>> df = pd.read_csv('data.csv')
        >>> model = train_prophet(df, col_target='energy_consumption')
    """
    
    logger.info(f"🔮 Training Prophet model for '{col_target}'...")
    
    # Prepare data in Prophet format
    if col_datetime not in df.columns:
        df_prophet = pd.DataFrame({
            'ds': df.index,
            'y': df[col_target].values
        })
    else:
        df_prophet = pd.DataFrame({
            'ds': df[col_datetime],
            'y': df[col_target].values
        })
    
    # Handle missing values
    df_prophet = df_prophet.dropna()
    df_prophet['ds'] = pd.to_datetime(df_prophet['ds'])
    
    logger.info(f"  Training data shape: {df_prophet.shape}")
    
    # Create and configure Prophet model
    model = Prophet(
        yearly_seasonality=PROPHET_YEARLY_SEASONALITY,
        weekly_seasonality=PROPHET_WEEKLY_SEASONALITY,
        daily_seasonality=PROPHET_DAILY_SEASONALITY,
        changepoint_prior_scale=PROPHET_CHANGEPOINT_PRIOR_SCALE,
        seasonality_scale=PROPHET_SEASONALITY_SCALE,
        interval_width=PROPHET_INTERVAL_WIDTH,
        interval_width_threshold=0.05
    )
    
    # Fit model (suppress verbose output)
    with open(os.devnull, 'w') as devnull:
        import sys
        old_stdout = sys.stdout
        sys.stdout = devnull
        model.fit(df_prophet)
        sys.stdout = old_stdout
    
    logger.info("✓ Prophet model trained successfully")
    
    # Save model if name provided
    if model_name:
        model_path = os.path.join(MODELS_PATH, f'{model_name}_prophet.pkl')
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        logger.info(f"  Model saved to: {model_path}")
    
    return model


def forecast_prophet(model, periods=24):
    """
    Generate forecasts using trained Prophet model
    
    Args:
        model: Trained Prophet model
        periods (int): Number of periods to forecast
        
    Returns:
        pd.DataFrame: Forecast dataframe with 'yhat', 'yhat_lower', 'yhat_upper'
        
    Example:
        >>> forecast = forecast_prophet(model, periods=24)
        >>> print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']])
    """
    
    logger.info(f"📈 Generating forecast for {periods} periods...")
    
    # Create future dataframe
    future = model.make_future_dataframe(periods=periods)
    
    # Generate forecast
    with open(os.devnull, 'w') as devnull:
        import sys
        old_stdout = sys.stdout
        sys.stdout = devnull
        forecast = model.predict(future)
        sys.stdout = old_stdout
    
    # Return only relevant columns
    forecast_result = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(periods)
    
    logger.info(f"✓ Forecast generated with {len(forecast_result)} periods")
    
    return forecast_result


def load_prophet_model(model_name):
    """
    Load trained Prophet model from file
    
    Args:
        model_name (str): Name of the saved model
        
    Returns:
        Prophet.model: Loaded Prophet model
    """
    model_path = os.path.join(MODELS_PATH, f'{model_name}_prophet.pkl')
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    logger.info(f"✓ Loaded Prophet model from: {model_path}")
    return model


def train_prophet_for_all_columns(df, exclude_cols=None, 
                                   col_datetime='ds', prefix='prophet'):
    """
    Train Prophet models for all numeric columns
    
    Args:
        df (pd.DataFrame): Input dataframe
        exclude_cols (list): Columns to exclude from modeling
        col_datetime (str): Datetime column name
        prefix (str): Prefix for saving models
        
    Returns:
        dict: Dictionary mapping column names to trained models
    """
    
    if exclude_cols is None:
        exclude_cols = []
    
    models = {}
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        if col not in exclude_cols:
            logger.info(f"Training model for column: {col}")
            model = train_prophet(
                df, col, col_datetime=col_datetime,
                model_name=f'{prefix}_{col}'
            )
            models[col] = model
    
    logger.info(f"✓ Trained {len(models)} Prophet models")
    return models


if __name__ == "__main__":
    print("✓ Prophet model module ready!")
    print("\nUsage example:")
    print("  from models.prophet_model import train_prophet, forecast_prophet")
    print("  model = train_prophet(df, col_target='your_column')")
    print("  forecast = forecast_prophet(model, periods=24)")

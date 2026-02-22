"""
Prediction Module
Make predictions using trained models on new data
"""

import numpy as np
import pandas as pd
from src.models.lstm_model import load_lstm_model, predict_lstm
from src.models.prophet_model import load_prophet_model, forecast_prophet
from src.data_preprocessing import DataPreprocessor
from src.config import MODELS_PATH, FORECAST_HORIZON, CONFIDENCE_LEVEL
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PredictionEngine:
    """Engine for making predictions with trained models"""
    
    def __init__(self, model_prefix):
        """
        Initialize prediction engine with trained models
        
        Args:
            model_prefix (str): Prefix of saved model files
        """
        self.model_prefix = model_prefix
        self.lstm_model = None
        self.prophet_models = {}
        self.scaler = None
        self.preprocessor = DataPreprocessor()
        
        self._load_models()
        logger.info("✓ PredictionEngine initialized")
    
    def _load_models(self):
        """Load LSTM and Prophet models"""
        try:
            self.lstm_model = load_lstm_model(self.model_prefix)
            logger.info("✓ LSTM model loaded")
        except FileNotFoundError:
            logger.warning("⚠️  LSTM model not found")
        
        # Try to load Prophet models by checking MODELS_PATH
        try:
            import pickle
            for filename in os.listdir(MODELS_PATH):
                if self.model_prefix in filename and filename.endswith('_prophet.pkl'):
                    with open(os.path.join(MODELS_PATH, filename), 'rb') as f:
                        col_name = filename.replace(f'{self.model_prefix}_', '').replace('_prophet.pkl', '')
                        self.prophet_models[col_name] = pickle.load(f)
            logger.info(f"✓ Loaded {len(self.prophet_models)} Prophet models")
        except Exception as e:
            logger.warning(f"⚠️  Could not load Prophet models: {e}")
    
    def predict_lstm(self, X_data, inverse_scale=True):
        """
        Make LSTM predictions
        
        Args:
            X_data (np.ndarray): Input sequences (batch_size, timesteps, features)
            inverse_scale (bool): Whether to inverse scale predictions
            
        Returns:
            np.ndarray: Predictions
            
        Example:
            >>> engine = PredictionEngine('my_model')
            >>> predictions = engine.predict_lstm(X_test)
        """
        
        if self.lstm_model is None:
            raise ValueError("LSTM model not loaded")
        
        logger.info("🤖 Making LSTM predictions...")
        predictions = predict_lstm(self.lstm_model, X_data)
        
        if inverse_scale and self.scaler is not None:
            predictions = self.scaler.inverse_transform(predictions)
        
        logger.info(f"✓ Generated {len(predictions)} predictions")
        return predictions
    
    def predict_prophet(self, col_name, periods=FORECAST_HORIZON):
        """
        Make Prophet predictions for a specific column
        
        Args:
            col_name (str): Column name
            periods (int): Number of periods to forecast
            
        Returns:
            pd.DataFrame: Forecast dataframe
        """
        
        if col_name not in self.prophet_models:
            raise ValueError(f"Prophet model not found for column: {col_name}")
        
        logger.info(f"🔮 Making Prophet predictions for '{col_name}'...")
        model = self.prophet_models[col_name]
        forecast = forecast_prophet(model, periods=periods)
        
        return forecast
    
    def predict_all(self, new_data, datetime_col=None, inverse_scale=True):
        """
        Make predictions with all available models
        
        Args:
            new_data: Either path to CSV file or pd.DataFrame
            datetime_col (str): Name of datetime column
            inverse_scale (bool): Whether to inverse scale
            
        Returns:
            dict: Results from all models
        """
        
        logger.info("📊 Making predictions with all models...")
        results = {}
        
        # Load data
        if isinstance(new_data, str):
            df = self.preprocessor.load_raw_data(new_data)
        else:
            df = new_data.copy()
        
        # Preprocess
        df = self.preprocessor.clean_data(df)
        df = self.preprocessor.create_features(df, datetime_col=datetime_col)
        
        # LSTM predictions
        if self.lstm_model is not None:
            try:
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                data = df[numeric_cols].values
                X_scaled = self.preprocessor.scaler.transform(data)
                
                # Reshape for LSTM (assuming single sequence)
                if len(X_scaled.shape) == 2:
                    X_lstm = X_scaled[np.newaxis, :, :]
                else:
                    X_lstm = X_scaled
                
                lstm_pred = self.predict_lstm(X_lstm, inverse_scale=inverse_scale)
                results['lstm'] = {
                    'predictions': lstm_pred,
                    'shape': lstm_pred.shape
                }
                logger.info("✓ LSTM predictions generated")
            except Exception as e:
                logger.warning(f"⚠️  LSTM prediction failed: {e}")
        
        # Prophet predictions
        prophet_results = {}
        for col_name, model in self.prophet_models.items():
            try:
                forecast = self.predict_prophet(col_name)
                prophet_results[col_name] = forecast
            except Exception as e:
                logger.warning(f"⚠️  Prophet prediction for {col_name} failed: {e}")
        
        if prophet_results:
            results['prophet'] = prophet_results
            logger.info(f"✓ Prophet predictions generated for {len(prophet_results)} columns")
        
        logger.info("✓ All predictions completed")
        return results
    
    def predict_from_csv(self, csv_filename, datetime_col=None):
        """
        Make predictions from CSV file
        
        Args:
            csv_filename (str): CSV filename in data/raw/
            datetime_col (str): Name of datetime column
            
        Returns:
            dict: Predictions from all models
        """
        logger.info(f"📁 Loading data from: {csv_filename}")
        return self.predict_all(csv_filename, datetime_col=datetime_col)


# ============================================================
# Convenience Functions
# ============================================================

def quick_predict(model_prefix, X_test, inverse_scale=True):
    """
    Quick prediction with LSTM
    
    Args:
        model_prefix (str): Model prefix
        X_test: Test data
        inverse_scale (bool): Inverse scale results
        
    Returns:
        np.ndarray: Predictions
    """
    engine = PredictionEngine(model_prefix)
    return engine.predict_lstm(X_test, inverse_scale=inverse_scale)


def predict_future(model_prefix, periods=FORECAST_HORIZON):
    """
    Predict future values using all models
    
    Args:
        model_prefix (str): Model prefix
        periods (int): Forecast horizon
        
    Returns:
        dict: Forecasts
    """
    engine = PredictionEngine(model_prefix)
    results = {}
    
    for col_name in engine.prophet_models.keys():
        results[col_name] = engine.predict_prophet(col_name, periods=periods)
    
    return results


if __name__ == "__main__":
    print("✓ Prediction module ready!")
    print("\nUsage example:")
    print("  from prediction import PredictionEngine")
    print("  engine = PredictionEngine('my_model')")
    print("  predictions = engine.predict_lstm(X_test)")

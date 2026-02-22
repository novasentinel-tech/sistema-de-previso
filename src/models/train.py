"""
Training Pipeline for TOTEM_DEEPSEA
Orchestrates training of all models
"""

import os
import numpy as np
import pandas as pd
from src.data_preprocessing import DataPreprocessor, quick_preprocess
from src.models.lstm_model import build_lstm_model, train_lstm, load_lstm_model
from src.models.prophet_model import train_prophet, train_prophet_for_all_columns
from src.config import MODELS_PATH, PROCESSED_PATH, RANDOM_SEED, LSTM_LOOKBACK
import logging
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TrainingPipeline:
    """Complete training pipeline for both LSTM and Prophet models"""
    
    def __init__(self):
        self.preprocessor = DataPreprocessor()
        self.lstm_model = None
        self.prophet_models = {}
        self.scaler = None
        self.metrics = {}
        logger.info("✓ TrainingPipeline initialized")
    
    def run_full_pipeline(self, data_filename, datetime_col=None, lookback=LSTM_LOOKBACK):
        """
        Run complete training pipeline from raw data to trained models
        
        Args:
            data_filename (str): CSV filename in data/raw/
            datetime_col (str): Name of datetime column (optional)
            lookback (int): Lookback window for sequences
            
        Returns:
            dict: Dictionary with training results
        """
        
        logger.info("="*60)
        logger.info("🚀 STARTING FULL TRAINING PIPELINE")
        logger.info("="*60)
        
        # Step 1: Load and preprocess data
        logger.info("\n📊 Step 1: Data Preprocessing")
        df = self.preprocessor.load_raw_data(data_filename)
        df = self.preprocessor.clean_data(df)
        df = self.preprocessor.create_features(df, datetime_col=datetime_col)
        
        # Step 2: Extract numeric data and normalize
        logger.info("\n📈 Step 2: Normalization")
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        data = df[numeric_cols].values
        X_train_scaled, X_test_scaled, X_val_scaled = self.preprocessor.normalize_data(data, data, data)
        self.scaler = self.preprocessor.scaler
        
        # Step 3: Create sequences for LSTM
        logger.info("\n🔗 Step 3: Sequence Creation")
        X, y = self.preprocessor.create_sequences(X_train_scaled, lookback=lookback)
        
        # Step 4: Train/test split
        logger.info("\n✂️  Step 4: Train/Test Split")
        X_train, X_val, X_test, y_train, y_val, y_test = \
            self.preprocessor.train_test_split(X, y)
        
        # Save processed data
        prefix = data_filename.replace('.csv', '')
        self.preprocessor.save_processed_data(
            X_train, X_val, X_test, y_train, y_val, y_test, prefix=prefix
        )
        
        # Step 5: Train LSTM model
        logger.info("\n🤖 Step 5: LSTM Training")
        self.lstm_model, lstm_history = train_lstm(
            X_train, y_train, X_val, y_val, model_name=prefix
        )
        
        # Step 6: Train Prophet models
        logger.info("\n🔮 Step 6: Prophet Training")
        self.prophet_models = train_prophet_for_all_columns(
            df, exclude_cols=[], col_datetime=datetime_col or 'ds',
            prefix=prefix
        )
        
        # Step 7: Save metrics
        logger.info("\n📊 Step 7: Saving Metrics")
        self.metrics = {
            'data_filename': data_filename,
            'total_samples': len(df),
            'features_count': len(numeric_cols),
            'lstm_final_loss': float(lstm_history.history['loss'][-1]),
            'lstm_final_val_loss': float(lstm_history.history['val_loss'][-1]),
            'train_samples': len(X_train),
            'val_samples': len(X_val),
            'test_samples': len(X_test),
            'prophet_models_count': len(self.prophet_models)
        }
        
        self._save_metrics(prefix)
        
        logger.info("\n" + "="*60)
        logger.info("✅ TRAINING PIPELINE COMPLETED SUCCESSFULLY")
        logger.info("="*60)
        
        return self.metrics
    
    def _save_metrics(self, prefix):
        """Save training metrics to JSON"""
        metrics_path = os.path.join(MODELS_PATH, f'{prefix}_metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(self.metrics, f, indent=4)
        logger.info(f"✓ Metrics saved to: {metrics_path}")
    
    def load_trained_models(self, prefix):
        """
        Load previously trained models
        
        Args:
            prefix (str): Prefix of model files
        """
        logger.info(f"Loading trained models with prefix: {prefix}")
        
        try:
            self.lstm_model = load_lstm_model(prefix)
            logger.info("✓ LSTM model loaded")
        except FileNotFoundError as e:
            logger.warning(f"⚠️  LSTM model not found: {e}")
        
        # Load Prophet models
        numeric_cols = self._get_numeric_cols_from_metrics(prefix)
        for col in numeric_cols:
            try:
                from models.prophet_model import load_prophet_model
                model = load_prophet_model(f'{prefix}_{col}')
                self.prophet_models[col] = model
            except:
                pass
        
        logger.info(f"✓ Loaded {len(self.prophet_models)} Prophet models")
    
    def _get_numeric_cols_from_metrics(self, prefix):
        """Get numeric columns from saved metrics"""
        metrics_path = os.path.join(MODELS_PATH, f'{prefix}_metrics.json')
        if os.path.exists(metrics_path):
            with open(metrics_path, 'r') as f:
                metrics = json.load(f)
                return list(range(metrics.get('features_count', 0)))
        return []


# ============================================================
# Convenience Functions
# ============================================================

def train_complete_system(data_filename, datetime_col=None):
    """
    Simple function to train complete system
    
    Args:
        data_filename (str): CSV filename in data/raw/
        datetime_col (str): Name of datetime column
        
    Returns:
        TrainingPipeline: Trained pipeline object
    """
    pipeline = TrainingPipeline()
    pipeline.run_full_pipeline(data_filename, datetime_col=datetime_col)
    return pipeline


if __name__ == "__main__":
    print("✓ Training module ready!")
    print("\nUsage example:")
    print("  from models.train import train_complete_system")
    print("  pipeline = train_complete_system('my_data.csv')")

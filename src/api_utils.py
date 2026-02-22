"""
API Utilities - Helper functions for data processing and model management
"""

import os
import json
import tempfile
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
import pickle
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FileManager:
    """Manages temporary storage of uploaded CSV files"""
    
    def __init__(self, storage_dir: str = None):
        """Initialize file manager with storage directory"""
        if storage_dir is None:
            storage_dir = os.path.join(tempfile.gettempdir(), "totem_deepsea_uploads")
        
        self.storage_dir = storage_dir
        os.makedirs(storage_dir, exist_ok=True)
        logger.info(f"✓ FileManager initialized at {storage_dir}")
    
    def save_csv(self, df: pd.DataFrame) -> str:
        """
        Save DataFrame to CSV and return file_id
        
        Args:
            df: pandas DataFrame
            
        Returns:
            file_id: unique identifier for stored file
        """
        file_id = str(uuid.uuid4())[:8]
        filepath = os.path.join(self.storage_dir, f"{file_id}.csv")
        
        df.to_csv(filepath, index=False)
        logger.info(f"✓ File saved: {file_id}")
        
        return file_id
    
    def load_csv(self, file_id: str) -> Optional[pd.DataFrame]:
        """
        Load CSV by file_id
        
        Args:
            file_id: unique file identifier
            
        Returns:
            pandas DataFrame or None if not found
        """
        filepath = os.path.join(self.storage_dir, f"{file_id}.csv")
        
        if not os.path.exists(filepath):
            logger.error(f"❌ File not found: {file_id}")
            return None
        
        df = pd.read_csv(filepath)
        logger.info(f"✓ File loaded: {file_id}")
        
        return df
    
    def delete_file(self, file_id: str) -> bool:
        """Delete file by file_id"""
        filepath = os.path.join(self.storage_dir, f"{file_id}.csv")
        
        if os.path.exists(filepath):
            os.remove(filepath)
            logger.info(f"✓ File deleted: {file_id}")
            return True
        
        return False


class ModelManager:
    """Manages storage and retrieval of trained models"""
    
    def __init__(self, models_dir: str = None):
        """Initialize model manager"""
        if models_dir is None:
            models_dir = os.path.join(tempfile.gettempdir(), "totem_deepsea_models")
        
        self.models_dir = models_dir
        os.makedirs(models_dir, exist_ok=True)
        self.models_registry = {}  # In-memory registry
        logger.info(f"✓ ModelManager initialized at {models_dir}")
    
    def save_model(self, model, model_id: str, metadata: Dict = None) -> bool:
        """
        Save model to disk
        
        Args:
            model: trained model object
            model_id: unique model identifier
            metadata: dictionary with model metadata
            
        Returns:
            bool: success status
        """
        try:
            model_path = os.path.join(self.models_dir, f"{model_id}.pkl")
            
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            
            # Save metadata
            if metadata:
                meta_path = os.path.join(self.models_dir, f"{model_id}_meta.json")
                with open(meta_path, 'w') as f:
                    json.dump(metadata, f)
            
            self.models_registry[model_id] = metadata or {}
            logger.info(f"✓ Model saved: {model_id}")
            
            return True
        except Exception as e:
            logger.error(f"❌ Error saving model: {e}")
            return False
    
    def load_model(self, model_id: str):
        """
        Load model from disk
        
        Args:
            model_id: unique model identifier
            
        Returns:
            tuple: (model, metadata) or (None, None) if not found
        """
        try:
            model_path = os.path.join(self.models_dir, f"{model_id}.pkl")
            
            if not os.path.exists(model_path):
                logger.error(f"❌ Model not found: {model_id}")
                return None, None
            
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            
            # Load metadata
            metadata = {}
            meta_path = os.path.join(self.models_dir, f"{model_id}_meta.json")
            if os.path.exists(meta_path):
                with open(meta_path, 'r') as f:
                    metadata = json.load(f)
            
            logger.info(f"✓ Model loaded: {model_id}")
            
            return model, metadata
        except Exception as e:
            logger.error(f"❌ Error loading model: {e}")
            return None, None
    
    def list_models(self) -> List[str]:
        """List all available models"""
        models = [f.replace('.pkl', '') for f in os.listdir(self.models_dir) 
                 if f.endswith('.pkl')]
        return models


class DataPreprocessor:
    """Data preprocessing utilities for API"""
    
    @staticmethod
    def validate_csv(df: pd.DataFrame) -> Tuple[bool, str, Optional[str], List[str]]:
        """
        Validate CSV structure and content
        
        Args:
            df: pandas DataFrame
            
        Returns:
            tuple: (is_valid, message, datetime_col, numeric_cols)
        """
        # Check if empty
        if df.empty:
            return False, "DataFrame is empty", None, []
        
        # Check for datetime column
        datetime_col = None
        for col in df.columns:
            if 'time' in col.lower() or 'date' in col.lower():
                datetime_col = col
                break
        
        # Get numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if not numeric_cols:
            return False, "No numeric columns found", None, []
        
        if len(df) < 50:
            return False, "Insufficient data (minimum 50 rows required)", datetime_col, numeric_cols
        
        return True, "✓ CSV validation passed", datetime_col, numeric_cols
    
    @staticmethod
    def generate_timestamps(df: pd.DataFrame, datetime_col: Optional[str], periods: int) -> List[str]:
        """
        Generate future timestamps for forecast
        
        Args:
            df: original DataFrame
            datetime_col: name of datetime column
            periods: number of future periods
            
        Returns:
            list: ISO format timestamps
        """
        timestamps = []
        
        if datetime_col and datetime_col in df.columns:
            try:
                # Parse last timestamp
                last_ts = pd.to_datetime(df[datetime_col].iloc[-1])
                
                # Infer frequency (daily by default)
                freq_timedelta = timedelta(days=1)
                if len(df) > 1:
                    freq = pd.to_datetime(df[datetime_col].iloc[-1]) - pd.to_datetime(df[datetime_col].iloc[-2])
                    freq_timedelta = freq
                
                # Generate future timestamps
                for i in range(1, periods + 1):
                    future_ts = last_ts + (freq_timedelta * i)
                    timestamps.append(future_ts.isoformat())
            except:
                # Fallback to default timestamps
                now = datetime.now()
                for i in range(periods):
                    ts = (now + timedelta(hours=i)).isoformat()
                    timestamps.append(ts)
        else:
            # Default timestamps
            now = datetime.now()
            for i in range(periods):
                ts = (now + timedelta(hours=i)).isoformat()
                timestamps.append(ts)
        
        return timestamps


class MetricsCalculator:
    """Calculate evaluation metrics"""
    
    @staticmethod
    def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Calculate MAE, RMSE, MAPE, R²
        
        Args:
            y_true: actual values
            y_pred: predicted values
            
        Returns:
            dict: metrics
        """
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        
        # Flatten if needed
        y_true = y_true.flatten() if isinstance(y_true, np.ndarray) else np.array(y_true)
        y_pred = y_pred.flatten() if isinstance(y_pred, np.ndarray) else np.array(y_pred)
        
        # Handle NaN/Inf
        y_true = np.nan_to_num(y_true, nan=0.0, posinf=1e6, neginf=-1e6)
        y_pred = np.nan_to_num(y_pred, nan=0.0, posinf=1e6, neginf=-1e6)
        
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        
        # MAPE with zero handling
        mape = np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + 1e-8))) * 100
        
        # R²
        try:
            r2 = r2_score(y_true, y_pred)
        except:
            r2 = None
        
        return {
            'mae': float(mae),
            'rmse': float(rmse),
            'mape': float(mape),
            'r2': float(r2) if r2 is not None else None
        }


class APIException(Exception):
    """Custom API exception"""
    
    def __init__(self, message: str, error_code: str, status_code: int = 400):
        self.message = message
        self.error_code = error_code
        self.status_code = status_code
        super().__init__(self.message)


# Global instances
file_manager = FileManager()
model_manager = ModelManager()

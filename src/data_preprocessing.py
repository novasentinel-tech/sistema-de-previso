"""
Data Preprocessing Module
Handles data loading, cleaning, normalization, and feature engineering
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from src.config import (
    RAW_PATH, PROCESSED_PATH, RANDOM_SEED,
    NORMALIZATION_METHOD, HANDLE_MISSING, REMOVE_OUTLIERS,
    OUTLIER_THRESHOLD, TEST_SIZE, VALIDATION_SIZE
)
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataPreprocessor:
    """Handles all data preprocessing operations"""
    
    def __init__(self, normalization_method=NORMALIZATION_METHOD):
        self.normalization_method = normalization_method
        self.scaler = None
        logger.info(f"DataPreprocessor initialized with {normalization_method} normalization")
    
    def load_raw_data(self, filename):
        """
        Load raw data from CSV file
        
        Args:
            filename (str): CSV filename in RAW_PATH
            
        Returns:
            pd.DataFrame: Loaded data
            
        Example:
            df = preprocessor.load_raw_data('energy_consumption.csv')
        """
        filepath = os.path.join(RAW_PATH, filename)
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")
        
        df = pd.read_csv(filepath)
        logger.info(f"✓ Loaded {len(df)} rows from {filename}")
        logger.info(f"  Columns: {list(df.columns)}")
        
        return df
    
    def clean_data(self, df):
        """
        Clean data by removing/filling NaNs and outliers
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            pd.DataFrame: Cleaned dataframe
        """
        logger.info("🧹 Starting data cleaning...")
        
        # Handle missing values
        initial_nulls = df.isnull().sum().sum()
        
        if HANDLE_MISSING == 'drop':
            df = df.dropna()
            logger.info(f"  Dropped {initial_nulls} null values")
            
        elif HANDLE_MISSING == 'forward_fill':
            df = df.ffill().bfill()
            logger.info(f"  Forward filled {initial_nulls} null values")
            
        elif HANDLE_MISSING == 'interpolate':
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            df[numeric_cols] = df[numeric_cols].interpolate(method='linear')
            logger.info(f"  Interpolated {initial_nulls} null values")
        # Remove outliers using IQR or Z-score
        if REMOVE_OUTLIERS:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
                outliers = (z_scores > OUTLIER_THRESHOLD).sum()
                if outliers > 0:
                    logger.info(f"  Removed {outliers} outliers from {col}")
                    df = df[z_scores <= OUTLIER_THRESHOLD]
        
        logger.info(f"✓ Data cleaning complete. Shape: {df.shape}")
        return df
    
    def create_features(self, df, datetime_col=None):
        """
        Create temporal features from datetime
        
        Args:
            df (pd.DataFrame): Input dataframe
            datetime_col (str): Name of datetime column (optional)
            
        Returns:
            pd.DataFrame: Dataframe with new features
        """
        logger.info("✨ Creating temporal features...")
        
        # Convert datetime if provided
        if datetime_col and datetime_col in df.columns:
            df[datetime_col] = pd.to_datetime(df[datetime_col])
            df = df.set_index(datetime_col)
        elif df.index.dtype == 'object':
            df.index = pd.to_datetime(df.index)
        
        # Extract temporal features
        df['hour'] = df.index.hour
        df['day_of_week'] = df.index.dayofweek
        df['month'] = df.index.month
        df['day'] = df.index.day
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        
        # Add rolling statistics for each numeric column
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col not in ['hour', 'day_of_week', 'month', 'day', 'is_weekend']:
                df[f'{col}_ma7'] = df[col].rolling(window=7, min_periods=1).mean()
                df[f'{col}_std7'] = df[col].rolling(window=7, min_periods=1).std()
        
        # Fill any NaN values that may have been created
        df = df.fillna(df.mean())
        df = df.ffill().bfill()
        
        logger.info(f"✓ Created {len(df.columns)} features total")
        return df
    
    def normalize_data(self, X_train, X_test=None, X_val=None):
        """
        Normalize data using MinMax or StandardScaler
        
        Args:
            X_train: Training data
            X_test: Test data (optional)
            X_val: Validation data (optional)
            
        Returns:
            tuple: Normalized X_train, X_test, X_val
        """
        logger.info(f"📊 Normalizing data using {self.normalization_method}...")
        
        # Clean data before normalization
        X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
        
        if self.normalization_method == 'minmax':
            self.scaler = MinMaxScaler(feature_range=(0, 1))
        elif self.normalization_method == 'standard':
            self.scaler = StandardScaler()
        else:
            raise ValueError(f"Unknown normalization method: {self.normalization_method}")
        
        # Fit on training data
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        results = [X_train_scaled]
        
        if X_test is not None:
            X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)
            X_test_scaled = self.scaler.transform(X_test)
            results.append(X_test_scaled)
        else:
            results.append(None)
        
        if X_val is not None:
            X_val = np.nan_to_num(X_val, nan=0.0, posinf=0.0, neginf=0.0)
            X_val_scaled = self.scaler.transform(X_val)
            results.append(X_val_scaled)
        else:
            results.append(None)
        
        logger.info("✓ Normalization complete")
        return tuple(results)
    
    def create_sequences(self, data, lookback=24):
        """
        Create sequences for LSTM model
        
        Args:
            data (np.ndarray): Normalized input data
            lookback (int): Number of timesteps to look back
            
        Returns:
            tuple: (X, y) sequences
        """
        X, y = [], []
        
        for i in range(len(data) - lookback):
            X.append(data[i:(i + lookback)])
            y.append(data[i + lookback])
        
        X = np.array(X)
        y = np.array(y)
        
        logger.info(f"✓ Created sequences: X.shape={X.shape}, y.shape={y.shape}")
        return X, y
    
    def train_test_split(self, X, y, test_size=TEST_SIZE, validation_size=VALIDATION_SIZE):
        """
        Split data into train, validation, test sets
        
        Args:
            X: Input features
            y: Target values
            test_size: Proportion for test set
            validation_size: Proportion for validation set
            
        Returns:
            tuple: (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        np.random.seed(RANDOM_SEED)
        
        total_samples = len(X)
        test_idx = int(total_samples * (1 - test_size))
        val_idx = int(test_idx * (1 - validation_size))
        
        X_train = X[:val_idx]
        y_train = y[:val_idx]
        
        X_val = X[val_idx:test_idx]
        y_val = y[val_idx:test_idx]
        
        X_test = X[test_idx:]
        y_test = y[test_idx:]
        
        logger.info(f"✓ Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def save_processed_data(self, X_train, X_val, X_test, y_train, y_val, y_test, prefix='default'):
        """
        Save processed data to files
        
        Args:
            X_train, X_val, X_test, y_train, y_val, y_test: Data arrays
            prefix (str): Prefix for filenames
        """
        np.save(os.path.join(PROCESSED_PATH, f'{prefix}_X_train.npy'), X_train)
        np.save(os.path.join(PROCESSED_PATH, f'{prefix}_X_val.npy'), X_val)
        np.save(os.path.join(PROCESSED_PATH, f'{prefix}_X_test.npy'), X_test)
        np.save(os.path.join(PROCESSED_PATH, f'{prefix}_y_train.npy'), y_train)
        np.save(os.path.join(PROCESSED_PATH, f'{prefix}_y_val.npy'), y_val)
        np.save(os.path.join(PROCESSED_PATH, f'{prefix}_y_test.npy'), y_test)
        
        logger.info(f"✓ Saved processed data with prefix '{prefix}' to {PROCESSED_PATH}")
    
    def load_processed_data(self, prefix='default'):
        """
        Load processed data from files
        
        Args:
            prefix (str): Prefix of filenames
            
        Returns:
            tuple: (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        X_train = np.load(os.path.join(PROCESSED_PATH, f'{prefix}_X_train.npy'))
        X_val = np.load(os.path.join(PROCESSED_PATH, f'{prefix}_X_val.npy'))
        X_test = np.load(os.path.join(PROCESSED_PATH, f'{prefix}_X_test.npy'))
        y_train = np.load(os.path.join(PROCESSED_PATH, f'{prefix}_y_train.npy'))
        y_val = np.load(os.path.join(PROCESSED_PATH, f'{prefix}_y_val.npy'))
        y_test = np.load(os.path.join(PROCESSED_PATH, f'{prefix}_y_test.npy'))
        
        logger.info(f"✓ Loaded processed data with prefix '{prefix}'")
        return X_train, X_val, X_test, y_train, y_val, y_test


# ============================================================
# Convenience functions
# ============================================================

def quick_preprocess(filename, lookback=24, datetime_col=None):
    """
    Quick preprocessing pipeline
    
    Args:
        filename (str): CSV filename in RAW_PATH
        lookback (int): Lookback window for sequences
        datetime_col (str): Name of datetime column
        
    Returns:
        tuple: Preprocessed train/val/test data and scaler
    """
    preprocessor = DataPreprocessor()
    
    # Load and clean
    df = preprocessor.load_raw_data(filename)
    df = preprocessor.clean_data(df)
    df = preprocessor.create_features(df, datetime_col=datetime_col)
    
    # Get numeric columns only
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    data = df[numeric_cols].values
    
    # Normalize
    X_train_scaled, X_test_scaled, X_val_scaled = preprocessor.normalize_data(data)
    
    # Create sequences
    X, y = preprocessor.create_sequences(X_train_scaled, lookback=lookback)
    
    # Split
    X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.train_test_split(X, y)
    
    return X_train, X_val, X_test, y_train, y_val, y_test, preprocessor.scaler


if __name__ == "__main__":
    print("✓ Data preprocessing module ready to use!")
    print("\nUsage example:")
    print("  from data_preprocessing import DataPreprocessor")
    print("  preprocessor = DataPreprocessor()")
    print("  df = preprocessor.load_raw_data('your_data.csv')")

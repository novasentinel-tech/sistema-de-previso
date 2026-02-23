import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from src.config import (
    RAW_PATH, PROCESSED_PATH, RANDOM_SEED,
    NORMALIZATION_METHOD, HANDLE_MISSING, REMOVE_OUTLIERS,
    OUTLIER_THRESHOLD, TEST_SIZE, VALIDATION_SIZE
)


class DataPreprocessor:
    
    def __init__(self, normalization_method=NORMALIZATION_METHOD):
        self.normalization_method = normalization_method
        self.scaler = None
    
    def load_raw_data(self, filename):
        filepath = os.path.join(RAW_PATH, filename)
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Arquivo não encontrado: {filepath}")
        return pd.read_csv(filepath)
    
    def clean_data(self, df):
        if HANDLE_MISSING == 'drop':
            df = df.dropna()
        elif HANDLE_MISSING == 'forward_fill':
            df = df.ffill().bfill()
        elif HANDLE_MISSING == 'interpolate':
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            df[numeric_cols] = df[numeric_cols].interpolate(method='linear')
        
        if REMOVE_OUTLIERS:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
                df = df[z_scores <= OUTLIER_THRESHOLD]
        
        return df
    
    def create_features(self, df, datetime_col=None):
        if datetime_col and datetime_col in df.columns:
            df[datetime_col] = pd.to_datetime(df[datetime_col])
            df = df.set_index(datetime_col)
        elif df.index.dtype == 'object':
            df.index = pd.to_datetime(df.index)
        
        df['hour'] = df.index.hour
        df['day_of_week'] = df.index.dayofweek
        df['month'] = df.index.month
        df['day'] = df.index.day
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col not in ['hour', 'day_of_week', 'month', 'day', 'is_weekend']:
                df[f'{col}_ma7'] = df[col].rolling(window=7, min_periods=1).mean()
                df[f'{col}_std7'] = df[col].rolling(window=7, min_periods=1).std()
        
        df = df.fillna(df.mean())
        df = df.ffill().bfill()
        return df
    
    def normalize_data(self, X_train, X_test=None, X_val=None):
        X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
        
        if self.normalization_method == 'minmax':
            self.scaler = MinMaxScaler(feature_range=(0, 1))
        elif self.normalization_method == 'standard':
            self.scaler = StandardScaler()
        else:
            raise ValueError(f"Método desconhecido: {self.normalization_method}")
        
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
        
        return tuple(results)
    
    def create_sequences(self, data, lookback=24):
        X, y = [], []
        for i in range(len(data) - lookback):
            X.append(data[i:(i + lookback)])
            y.append(data[i + lookback])
        return np.array(X), np.array(y)
    
    def train_test_split(self, X, y, test_size=TEST_SIZE, validation_size=VALIDATION_SIZE):
        np.random.seed(RANDOM_SEED)
        total_samples = len(X)
        test_idx = int(total_samples * (1 - test_size))
        val_idx = int(test_idx * (1 - validation_size))
        return X[:val_idx], X[val_idx:test_idx], X[test_idx:], y[:val_idx], y[val_idx:test_idx], y[test_idx:]
    
    def save_processed_data(self, X_train, X_val, X_test, y_train, y_val, y_test, prefix='default'):
        np.save(os.path.join(PROCESSED_PATH, f'{prefix}_X_train.npy'), X_train)
        np.save(os.path.join(PROCESSED_PATH, f'{prefix}_X_val.npy'), X_val)
        np.save(os.path.join(PROCESSED_PATH, f'{prefix}_X_test.npy'), X_test)
        np.save(os.path.join(PROCESSED_PATH, f'{prefix}_y_train.npy'), y_train)
        np.save(os.path.join(PROCESSED_PATH, f'{prefix}_y_val.npy'), y_val)
        np.save(os.path.join(PROCESSED_PATH, f'{prefix}_y_test.npy'), y_test)
    
    def load_processed_data(self, prefix='default'):
        return (
            np.load(os.path.join(PROCESSED_PATH, f'{prefix}_X_train.npy')),
            np.load(os.path.join(PROCESSED_PATH, f'{prefix}_X_val.npy')),
            np.load(os.path.join(PROCESSED_PATH, f'{prefix}_X_test.npy')),
            np.load(os.path.join(PROCESSED_PATH, f'{prefix}_y_train.npy')),
            np.load(os.path.join(PROCESSED_PATH, f'{prefix}_y_val.npy')),
            np.load(os.path.join(PROCESSED_PATH, f'{prefix}_y_test.npy'))
        )


def quick_preprocess(filename, lookback=24, datetime_col=None):
    preprocessor = DataPreprocessor()
    df = preprocessor.load_raw_data(filename)
    df = preprocessor.clean_data(df)
    df = preprocessor.create_features(df, datetime_col=datetime_col)
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    data = df[numeric_cols].values
    
    X_train_scaled, X_test_scaled, X_val_scaled = preprocessor.normalize_data(data)
    X, y = preprocessor.create_sequences(X_train_scaled, lookback=lookback)
    X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.train_test_split(X, y)
    
    return X_train, X_val, X_test, y_train, y_val, y_test, preprocessor.scaler

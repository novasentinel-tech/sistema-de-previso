"""
LSTM Model for Multivariate Time Series Forecasting
"""

import numpy as np
import os
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from src.config import (
    LSTM_UNITS, LSTM_DROPOUT, LSTM_DENSE_UNITS,
    LSTM_BATCH_SIZE, LSTM_EPOCHS, LSTM_LEARNING_RATE,
    LSTM_EARLY_STOPPING, LSTM_PATIENCE, MODELS_PATH, RANDOM_SEED
)
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set seeds for reproducibility
np.random.seed(RANDOM_SEED)
keras.utils.set_random_seed(RANDOM_SEED)


def build_lstm_model(input_shape):
    """
    Build and compile LSTM model for multivariate time series
    
    Args:
        input_shape (tuple): Shape of input data (timesteps, features)
        
    Returns:
        keras.Model: Compiled LSTM model
        
    Example:
        >>> model = build_lstm_model((24, 5))  # 24 timesteps, 5 features
        >>> print(model.summary())
    """
    
    logger.info(f"🏗️  Building LSTM model with input shape: {input_shape}")
    
    model = keras.Sequential(name='LSTM_Model')
    
    # Input layer
    model.add(layers.Input(shape=input_shape))
    
    # First LSTM layer with return sequences
    model.add(layers.LSTM(
        units=LSTM_UNITS[0],
        return_sequences=True,
        name='lstm_layer_1'
    ))
    model.add(layers.Dropout(LSTM_DROPOUT))
    
    # Second LSTM layer
    model.add(layers.LSTM(
        units=LSTM_UNITS[1],
        return_sequences=False,
        name='lstm_layer_2'
    ))
    model.add(layers.Dropout(LSTM_DROPOUT))
    
    # Dense layers
    model.add(layers.Dense(
        units=LSTM_DENSE_UNITS,
        activation='relu',
        name='dense_layer_1'
    ))
    model.add(layers.Dropout(LSTM_DROPOUT))
    
    # Output layer (matches number of features)
    num_features = input_shape[1]
    model.add(layers.Dense(
        units=num_features,
        activation='linear',
        name='output_layer'
    ))
    
    # Compile model
    optimizer = keras.optimizers.Adam(learning_rate=LSTM_LEARNING_RATE)
    model.compile(
        optimizer=optimizer,
        loss='mse',
        metrics=['mae', 'mse']
    )
    
    logger.info("✓ LSTM model built successfully")
    logger.info(f"  Total parameters: {model.count_params():,}")
    
    return model


def train_lstm(X_train, y_train, X_val, y_val, model_name='lstm_model'):
    """
    Build and train LSTM model
    
    Args:
        X_train (np.ndarray): Training input features
        y_train (np.ndarray): Training target values
        X_val (np.ndarray): Validation input features
        y_val (np.ndarray): Validation target values
        model_name (str): Name for saving the model
        
    Returns:
        tuple: (trained_model, history)
        
    Example:
        >>> model, history = train_lstm(X_train, y_train, X_val, y_val)
        >>> print(f"Final validation MAE: {history.history['val_mae'][-1]:.4f}")
    """
    
    logger.info("🚀 Starting LSTM training...")
    logger.info(f"  X_train shape: {X_train.shape}")
    logger.info(f"  y_train shape: {y_train.shape}")
    
    # Ensure data doesn't contain NaN and is clipped
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=1.0, neginf=0.0)
    X_val = np.nan_to_num(X_val, nan=0.0, posinf=1.0, neginf=0.0)
    y_train = np.nan_to_num(y_train, nan=0.0, posinf=1.0, neginf=0.0)
    y_val = np.nan_to_num(y_val, nan=0.0, posinf=1.0, neginf=0.0)
    
    # Clip values to reasonable range
    X_train = np.clip(X_train, -1e6, 1e6)
    X_val = np.clip(X_val, -1e6, 1e6)
    y_train = np.clip(y_train, -1e6, 1e6)
    y_val = np.clip(y_val, -1e6, 1e6)
    
    # Build model
    input_shape = (X_train.shape[1], X_train.shape[2])
    model = build_lstm_model(input_shape)
    
    # Define callbacks
    callbacks = []
    
    if LSTM_EARLY_STOPPING:
        early_stop = EarlyStopping(
            monitor='val_loss',
            patience=LSTM_PATIENCE,
            restore_best_weights=True,
            verbose=1
        )
        callbacks.append(early_stop)
    
    # Model checkpoint to save best weights
    model_path = os.path.join(MODELS_PATH, f'{model_name}_best.h5')
    checkpoint = ModelCheckpoint(
        model_path,
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    )
    callbacks.append(checkpoint)
    
    # Train model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=LSTM_EPOCHS,
        batch_size=LSTM_BATCH_SIZE,
        callbacks=callbacks,
        verbose=1
    )
    
    logger.info("✓ LSTM training completed")
    logger.info(f"  Final train loss: {history.history['loss'][-1]:.6f}")
    logger.info(f"  Final val loss: {history.history['val_loss'][-1]:.6f}")
    logger.info(f"  Model saved to: {model_path}")
    
    return model, history


def load_lstm_model(model_name='lstm_model'):
    """
    Load trained LSTM model
    
    Args:
        model_name (str): Name of the model
        
    Returns:
        keras.Model: Loaded model
    """
    model_path = os.path.join(MODELS_PATH, f'{model_name}_best.h5')
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    model = keras.models.load_model(model_path)
    logger.info(f"✓ Loaded LSTM model from: {model_path}")
    
    return model


def predict_lstm(model, X_data):
    """
    Make predictions using LSTM model
    
    Args:
        model: Keras LSTM model
        X_data (np.ndarray): Input data for prediction
        
    Returns:
        np.ndarray: Predictions
    """
    predictions = model.predict(X_data, verbose=0)
    return predictions


if __name__ == "__main__":
    print("✓ LSTM model module ready!")
    print("\nUsage example:")
    print("  from models.lstm_model import build_lstm_model, train_lstm")
    print("  model, history = train_lstm(X_train, y_train, X_val, y_val)")

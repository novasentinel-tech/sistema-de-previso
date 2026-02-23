import numpy as np
import os

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False


class LSTMModel:
    
    def __init__(self, input_shape, output_dim=1, dropout_rate=0.2, lstm_units=64):
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow não está instalado")
        
        self.input_shape = input_shape
        self.output_dim = output_dim
        self.dropout_rate = dropout_rate
        self.lstm_units = lstm_units
        self.model = None
        self.history = None
        self.scaler = None
    
    def build(self):
        self.model = Sequential([
            LSTM(self.lstm_units, activation='relu', input_shape=self.input_shape, return_sequences=True),
            Dropout(self.dropout_rate),
            LSTM(self.lstm_units // 2, activation='relu', return_sequences=False),
            Dropout(self.dropout_rate),
            Dense(32, activation='relu'),
            Dense(self.output_dim)
        ])
        
        self.model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
        return self.model
    
    def train(self, X_train, y_train, X_val=None, y_val=None, epochs=50, batch_size=32):
        if self.model is None:
            self.build()
        
        callbacks = [EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)]
        
        if X_val is not None and y_val is not None:
            self.history = self.model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=epochs,
                batch_size=batch_size,
                callbacks=callbacks,
                verbose=0
            )
        else:
            self.history = self.model.fit(
                X_train, y_train,
                epochs=epochs,
                batch_size=batch_size,
                callbacks=callbacks,
                verbose=0
            )
        
        return self.history
    
    def predict(self, X):
        if self.model is None:
            raise ValueError("Modelo não foi treinado")
        return self.model.predict(X, verbose=0)
    
    def evaluate(self, X_test, y_test):
        if self.model is None:
            raise ValueError("Modelo não foi treinado")
        loss, mae = self.model.evaluate(X_test, y_test, verbose=0)
        return {'loss': loss, 'mae': mae}
    
    def save(self, filepath):
        if self.model is None:
            raise ValueError("Modelo não foi treinado")
        self.model.save(filepath)
    
    def load(self, filepath):
        self.model = tf.keras.models.load_model(filepath)
        return self.model
    
    def get_model_summary(self):
        if self.model is None:
            return "Modelo não foi construído"
        self.model.summary()


def build_lstm_model(input_shape, lstm_units=64, dropout_rate=0.2):
    lstm = LSTMModel(input_shape=input_shape, lstm_units=lstm_units, dropout_rate=dropout_rate)
    return lstm.build()


def train_lstm(X_train, y_train, X_val, y_val, model_name='lstm', epochs=50, batch_size=32):
    input_shape = (X_train.shape[1], X_train.shape[2])
    lstm = LSTMModel(input_shape=input_shape, lstm_units=64)
    lstm.build()
    
    history = lstm.train(
        X_train, y_train,
        X_val=X_val, y_val=y_val,
        epochs=epochs,
        batch_size=batch_size
    )
    
    return lstm.model, history


def load_lstm_model(filepath):
    lstm = LSTMModel(input_shape=(1, 1))
    return lstm.load(filepath)

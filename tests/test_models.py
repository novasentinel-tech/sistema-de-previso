"""
Tests for Model Training and Prediction
"""

import pytest
import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from models.lstm_model import build_lstm_model, predict_lstm


class TestLSTMModel:
    """Tests for LSTM model"""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing"""
        np.random.seed(42)
        X_train = np.random.randn(50, 24, 3)  # 50 samples, 24 timesteps, 3 features
        y_train = np.random.randn(50, 3)
        X_val = np.random.randn(10, 24, 3)
        y_val = np.random.randn(10, 3)
        X_test = np.random.randn(10, 24, 3)
        y_test = np.random.randn(10, 3)
        
        return X_train, y_train, X_val, y_val, X_test, y_test
    
    def test_build_model(self):
        """Test LSTM model building"""
        input_shape = (24, 3)  # 24 timesteps, 3 features
        model = build_lstm_model(input_shape)
        
        assert model is not None
        assert model.input_shape == (None, 24, 3)
        assert model.output_shape == (None, 3)
    
    def test_model_prediction(self, sample_data):
        """Test model prediction"""
        _, _, _, _, X_test, _ = sample_data
        
        input_shape = (24, 3)
        model = build_lstm_model(input_shape)
        
        # Make predictions
        predictions = predict_lstm(model, X_test)
        
        assert predictions.shape == (X_test.shape[0], 3)
        assert not np.isnan(predictions).any()
    
    def test_model_output_shape(self, sample_data):
        """Test that model output matches expected shape"""
        X_train, y_train, _, _, _, _ = sample_data
        
        input_shape = (X_train.shape[1], X_train.shape[2])
        model = build_lstm_model(input_shape)
        
        predictions = model.predict(X_train[:5], verbose=0)
        
        assert predictions.shape[0] == 5
        assert predictions.shape[1] == input_shape[1]


class TestProphetModel:
    """Tests for Prophet model"""
    
    def test_prophet_import(self):
        """Test if Prophet can be imported"""
        try:
            from prophet import Prophet
            assert Prophet is not None
        except ImportError:
            pytest.skip("Prophet not installed")
    
    def test_prophet_basic_training(self):
        """Test basic Prophet model training"""
        pytest.importorskip("prophet")
        
        import pandas as pd
        from models.prophet_model import train_prophet
        
        # Create simple time series
        dates = pd.date_range('2024-01-01', periods=100)
        data = pd.DataFrame({
            'ds': dates,
            'y': np.random.randn(100) + 10
        })
        data.set_index('ds', inplace=True)
        
        # Train model
        model = train_prophet(data, col_target='y', col_datetime='ds')
        
        assert model is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

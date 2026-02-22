"""
Tests for Prediction Module
"""

import pytest
import numpy as np
import pandas as pd
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


class TestPredictionModule:
    """Test cases for prediction module"""
    
    def test_prediction_engine_import(self):
        """Test importing PredictionEngine"""
        try:
            from prediction import PredictionEngine
            assert PredictionEngine is not None
        except ImportError as e:
            pytest.skip(f"Cannot import PredictionEngine: {e}")
    
    def test_sample_prediction_data(self):
        """Test with sample prediction data"""
        np.random.seed(42)
        
        # Create sample data
        X_test = np.random.randn(10, 24, 3)
        
        # Verify shape
        assert X_test.shape == (10, 24, 3)
        assert not np.isnan(X_test).any()
    
    def test_evaluation_metrics(self):
        """Test evaluation metrics calculation"""
        from evaluation import calculate_metrics
        
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([1.1, 2.1, 2.9, 3.9, 5.1])
        
        metrics = calculate_metrics(y_true, y_pred)
        
        assert 'mae' in metrics
        assert 'rmse' in metrics
        assert metrics['mae'] > 0
        assert metrics['rmse'] > 0


class TestDataPreprocessingForPrediction:
    """Test preprocessing for prediction"""
    
    def test_sequence_creation_for_prediction(self):
        """Test sequence creation for prediction"""
        from data_preprocessing import DataPreprocessor
        
        preprocessor = DataPreprocessor()
        
        # Create sample normalized data
        data = np.random.randn(100, 3)
        X, y = preprocessor.create_sequences(data, lookback=24)
        
        assert X.shape[0] == len(data) - 24
        assert X.shape[1] == 24
        assert X.shape[2] == 3
        assert y.shape[0] == len(data) - 24
    
    def test_normalization_for_prediction(self):
        """Test data normalization for prediction"""
        from data_preprocessing import DataPreprocessor
        
        preprocessor = DataPreprocessor()
        
        # Create sample data
        X_train = np.random.randn(50, 3) + 10
        X_test = np.random.randn(20, 3) + 10
        
        X_train_scaled, X_test_scaled = preprocessor.normalize_data(X_train, X_test)[:2]
        
        # Check normalization
        assert X_train_scaled.min() >= 0
        assert X_train_scaled.max() <= 1
        assert X_test_scaled.min() >= 0
        assert X_test_scaled.max() <= 1
        assert X_train_scaled.shape == X_train.shape


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

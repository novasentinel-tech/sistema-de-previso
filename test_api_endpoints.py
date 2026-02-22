"""
Test FastAPI endpoints
"""

import pytest
import pandas as pd
import numpy as np
from io import BytesIO
from fastapi.testclient import TestClient
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from main import app, active_files, active_models

client = TestClient(app)


class TestHealthCheck:
    """Test health check endpoint"""
    
    def test_health_check(self):
        """Test /health endpoint"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert "version" in data


class TestUploadCSV:
    """Test CSV upload endpoint"""
    
    def test_valid_csv_upload(self):
        """Test uploading valid CSV"""
        # Create test CSV
        df = pd.DataFrame({
            'date': pd.date_range('2020-01-01', periods=100),
            'value1': np.random.randn(100),
            'value2': np.random.randn(100)
        })
        
        csv_content = df.to_csv(index=False)
        files = {'file': ('test.csv', BytesIO(csv_content.encode()), 'text/csv')}
        
        response = client.post("/upload_csv", files=files)
        assert response.status_code == 200
        
        data = response.json()
        assert "file_id" in data
        assert data["rows"] == 100
        assert "date" in data["columns"]
        assert "value1" in data["numeric_columns"]
        assert "value2" in data["numeric_columns"]
    
    def test_csv_too_small(self):
        """Test CSV with insufficient data"""
        df = pd.DataFrame({
            'date': pd.date_range('2020-01-01', periods=10),
            'value': np.random.randn(10)
        })
        
        csv_content = df.to_csv(index=False)
        files = {'file': ('test.csv', BytesIO(csv_content.encode()), 'text/csv')}
        
        response = client.post("/upload_csv", files=files)
        assert response.status_code == 400
        data = response.json()
        assert "error" in data


class TestTraining:
    """Test model training endpoints"""
    
    @pytest.fixture
    def file_id(self):
        """Upload test file"""
        df = pd.DataFrame({
            'date': pd.date_range('2020-01-01', periods=100),
            'value1': np.random.randn(100),
            'value2': np.random.randn(100)
        })
        
        csv_content = df.to_csv(index=False)
        files = {'file': ('test.csv', BytesIO(csv_content.encode()), 'text/csv')}
        
        response = client.post("/upload_csv", files=files)
        return response.json()["file_id"]
    
    def test_lstm_training(self, file_id):
        """Test LSTM training"""
        request_data = {
            "file_id": file_id,
            "lookback": 10,
            "epochs": 2,
            "batch_size": 16
        }
        
        response = client.post("/train_lstm", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        assert "model_id" in data
        assert data["model_type"] == "lstm"
        assert data["rows_used"] > 0
        assert "training_time" in data
    
    def test_prophet_training(self, file_id):
        """Test Prophet training"""
        request_data = {"file_id": file_id}
        
        response = client.post("/train_prophet", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        assert "model_id" in data
        assert data["model_type"] == "prophet"
        assert "training_time" in data


class TestForecasting:
    """Test forecasting endpoints"""
    
    @pytest.fixture
    def lstm_model_id(self):
        """Train an LSTM model"""
        # Upload file
        df = pd.DataFrame({
            'date': pd.date_range('2020-01-01', periods=100),
            'value1': np.random.randn(100),
            'value2': np.random.randn(100)
        })
        
        csv_content = df.to_csv(index=False)
        files = {'file': ('test.csv', BytesIO(csv_content.encode()), 'text/csv')}
        
        response = client.post("/upload_csv", files=files)
        file_id = response.json()["file_id"]
        
        # Train model
        request_data = {
            "file_id": file_id,
            "lookback": 10,
            "epochs": 2,
            "batch_size": 16
        }
        
        response = client.post("/train_lstm", json=request_data)
        return response.json()["model_id"]
    
    def test_lstm_forecast(self, lstm_model_id):
        """Test LSTM forecasting"""
        response = client.get(
            "/forecast_lstm",
            params={"model_id": lstm_model_id, "periods": 24}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "forecast" in data
        assert "timestamps" in data
        assert data["model_type"] == "lstm"
        assert data["periods"] == 24
        assert len(data["timestamps"]) == 24


class TestManagement:
    """Test model and file management endpoints"""
    
    def test_list_files(self):
        """Test /files endpoint"""
        response = client.get("/files")
        assert response.status_code == 200
        data = response.json()
        assert "total" in data
        assert "files" in data
    
    def test_list_models(self):
        """Test /models endpoint"""
        response = client.get("/models")
        assert response.status_code == 200
        data = response.json()
        assert "total" in data
        assert "models" in data


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

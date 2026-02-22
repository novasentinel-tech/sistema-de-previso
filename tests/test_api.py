"""
Testes para API FastAPI TOTEM_DEEPSEA
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
import os
from fastapi.testclient import TestClient
from io import BytesIO
import sys

# Adicionar src ao path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Importar a aplicação
from api import app, data_manager

client = TestClient(app)


# ============================================================
# FIXTURES
# ============================================================

@pytest.fixture
def sample_csv_data():
    """Criar CSV de exemplo para testes"""
    dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
    data = {
        'Date': dates,
        'Close': np.random.normal(100, 10, 100),
        'Volume': np.random.randint(1000, 10000, 100)
    }
    df = pd.DataFrame(data)
    
    # Converter para BytesIO
    buffer = BytesIO()
    df.to_csv(buffer, index=False)
    buffer.seek(0)
    
    return buffer, 'test_data.csv'


@pytest.fixture
def sample_csv_file(sample_csv_data):
    """Criar arquivo CSV para upload"""
    buffer, filename = sample_csv_data
    return buffer, filename


# ============================================================
# TESTES DE HEALTH CHECK
# ============================================================

def test_root_endpoint():
    """Testar endpoint raiz"""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "✅ online"
    assert "version" in data


def test_health_endpoint():
    """Testar health check"""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "✅ Healthy"


# ============================================================
# TESTES DE UPLOAD
# ============================================================

def test_upload_csv_success(sample_csv_file):
    """Testar upload bem-sucedido de CSV"""
    buffer, filename = sample_csv_file
    
    response = client.post(
        "/upload_csv",
        files={"file": (filename, buffer, "text/csv")}
    )
    
    assert response.status_code == 200
    data = response.json()
    assert data["success"] == True
    assert data["filename"] == filename
    assert data["shape"]["rows"] == 100


def test_upload_csv_invalid_format():
    """Testar upload com formato inválido"""
    buffer = BytesIO(b"some invalid data")
    
    response = client.post(
        "/upload_csv",
        files={"file": ("test.txt", buffer, "text/plain")}
    )
    
    # Deve rejeitar arquivos não CSV
    assert response.status_code == 400


def test_upload_csv_empty():
    """Testar upload com CSV vazio"""
    buffer = BytesIO(b"col1,col2\n")  # Apenas cabeçalho
    
    response = client.post(
        "/upload_csv",
        files={"file": ("empty.csv", buffer, "text/csv")}
    )
    
    assert response.status_code == 400


# ============================================================
# TESTES DE LISTAGEM
# ============================================================

def test_list_uploads(sample_csv_file):
    """Testar listagem de arquivos enviados"""
    # Upload primeiro
    buffer, filename = sample_csv_file
    
    client.post(
        "/upload_csv",
        files={"file": (filename, buffer, "text/csv")}
    )
    
    # Listar
    response = client.get("/uploads")
    assert response.status_code == 200
    data = response.json()
    assert data["success"] == True
    assert data["total_uploads"] >= 1


def test_list_models():
    """Testar listagem de modelos"""
    response = client.get("/models")
    assert response.status_code == 200
    data = response.json()
    assert data["success"] == True
    assert "models" in data


# ============================================================
# TESTES DE TREINAMENTO
# ============================================================

def test_train_lstm_success(sample_csv_file):
    """Testar treinamento LSTM bem-sucedido"""
    # Upload primeiro
    buffer, filename = sample_csv_file
    
    client.post(
        "/upload_csv",
        files={"file": (filename, buffer, "text/csv")}
    )
    
    # Treinar
    response = client.post(
        f"/train_lstm?filename={filename}",
        json={
            "lookback": 10,
            "epochs": 5,
            "batch_size": 8,
            "test_size": 0.2
        }
    )
    
    assert response.status_code == 200
    data = response.json()
    assert data["success"] == True
    assert "model_name" in data
    assert "test_metrics" in data


def test_train_prophet_success(sample_csv_file):
    """Testar treinamento Prophet bem-sucedido"""
    # Upload primeiro
    buffer, filename = sample_csv_file
    
    client.post(
        "/upload_csv",
        files={"file": (filename, buffer, "text/csv")}
    )
    
    # Treinar
    response = client.post(
        f"/train_prophet?filename={filename}&column_to_forecast=Close"
    )
    
    assert response.status_code == 200
    data = response.json()
    assert data["success"] == True
    assert "model_name" in data


def test_train_lstm_invalid_file():
    """Testar treinamento com arquivo inexistente"""
    response = client.post(
        "/train_lstm?filename=nonexistent.csv"
    )
    
    assert response.status_code == 404


# ============================================================
# TESTES DE PREVISÃO
# ============================================================

def test_forecast_lstm_after_training(sample_csv_file):
    """Testar previsão LSTM após treinamento"""
    # Upload e treinar
    buffer, filename = sample_csv_file
    
    client.post(
        "/upload_csv",
        files={"file": (filename, buffer, "text/csv")}
    )
    
    client.post(
        f"/train_lstm?filename={filename}",
        json={"lookback": 10, "epochs": 5, "batch_size": 8}
    )
    
    # Fazer previsão
    response = client.get(
        f"/forecast_lstm?filename={filename}&periods=24"
    )
    
    assert response.status_code == 200
    data = response.json()
    assert data["success"] == True
    assert len(data["forecast"]) == 24
    assert len(data["timestamps"]) == 24
    assert "metrics" in data


def test_forecast_prophet_after_training(sample_csv_file):
    """Testar previsão Prophet após treinamento"""
    # Upload e treinar
    buffer, filename = sample_csv_file
    
    client.post(
        "/upload_csv",
        files={"file": (filename, buffer, "text/csv")}
    )
    
    client.post(
        f"/train_prophet?filename={filename}&column_to_forecast=Close"
    )
    
    # Fazer previsão
    response = client.get(
        f"/forecast_prophet?filename={filename}&periods=24"
    )
    
    assert response.status_code == 200
    data = response.json()
    assert data["success"] == True
    assert len(data["forecast"]) >= 1
    assert "metrics" in data


def test_forecast_invalid_model():
    """Testar previsão com modelo inexistente"""
    response = client.get(
        "/forecast_lstm?filename=nonexistent.csv&periods=24"
    )
    
    assert response.status_code == 404


# ============================================================
# TESTES DE VALIDAÇÃO
# ============================================================

def test_csv_column_validation(sample_csv_file):
    """Testar validação de colunas CSV"""
    buffer, filename = sample_csv_file
    
    # Upload com sucesso deve validar colunas
    response = client.post(
        "/upload_csv",
        files={"file": (filename, buffer, "text/csv")}
    )
    
    assert response.status_code == 200
    data = response.json()
    assert len(data["columns"]) >= 2


def test_response_format_lstm(sample_csv_file):
    """Testar formato de resposta LSTM"""
    buffer, filename = sample_csv_file
    
    client.post(
        "/upload_csv",
        files={"file": (filename, buffer, "text/csv")}
    )
    
    client.post(
        f"/train_lstm?filename={filename}",
        json={"lookback": 10, "epochs": 5}
    )
    
    response = client.get(
        f"/forecast_lstm?filename={filename}&periods=10"
    )
    
    data = response.json()
    
    # Validar estrutura
    assert "forecast" in data
    assert "timestamps" in data
    assert "metrics" in data
    assert "message" in data
    assert "success" in data
    assert data["metrics"]["model_type"] == "LSTM"


# ============================================================
# TESTES DE PERÍODOS
# ============================================================

def test_forecast_different_periods(sample_csv_file):
    """Testar previsão com diferentes períodos"""
    buffer, filename = sample_csv_file
    
    client.post(
        "/upload_csv",
        files={"file": (filename, buffer, "text/csv")}
    )
    
    client.post(
        f"/train_lstm?filename={filename}",
        json={"lookback": 10, "epochs": 5}
    )
    
    # Testar período 12
    response = client.get(f"/forecast_lstm?filename={filename}&periods=12")
    assert len(response.json()["forecast"]) == 12
    
    # Testar período 48
    response = client.get(f"/forecast_lstm?filename={filename}&periods=48")
    assert len(response.json()["forecast"]) == 48


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

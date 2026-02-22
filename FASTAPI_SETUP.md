# 🚀 TOTEM_DEEPSEA FastAPI - Quick Setup Guide

## Iniciando a API

### 1. Verificar Environment Python

```bash
# Confirmar que o venv está ativo
which python
python --version  # Deve ser 3.12+

# Ativar venv se necessário
source venv/bin/activate
```

### 2. Iniciar Servidor FastAPI

```bash
# Opção 1: Development com reload automático
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload

# Opção 2: Production com Gunicorn
gunicorn main:app --workers 4 --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

### 3. Acessar a API

- **API Base**: http://localhost:8000
- **Swagger API Docs**: http://localhost:8000/docs
- **ReDoc Docs**: http://localhost:8000/redoc
- **Health Check**: http://localhost:8000/health

## 📝 Workflow Básico

### 1. Upload de Dados

```bash
curl -X POST "http://localhost:8000/upload_csv" \
  -F "file=@data.csv"
```

Resposta:
```json
{
  "file_id": "a1b2c3d4",
  "rows": 365,
  "columns": ["date", "close", "volume"],
  "datetime_column": "date",
  "numeric_columns": ["close", "volume"]
}
```

### 2. Treinar LSTM

```bash
curl -X POST "http://localhost:8000/train_lstm" \
  -H "Content-Type: application/json" \
  -d '{
    "file_id": "a1b2c3d4",
    "lookback": 30,
    "epochs": 50,
    "batch_size": 16
  }'
```

Resposta:
```json
{
  "model_id": "lstm_a1b2c3d4_20240115_103045",
  "model_type": "lstm",
  "training_time": 45.321
}
```

### 3. Fazer Previsão

```bash
curl "http://localhost:8000/forecast_lstm?model_id=lstm_a1b2c3d4_20240115_103045&periods=24"
```

## 📊 Python Client Example

```python
import requests
import pandas as pd

BASE_URL = "http://localhost:8000"

# Upload
with open('data.csv', 'rb') as f:
    r = requests.post(f"{BASE_URL}/upload_csv", files={'file': f})
    file_id = r.json()['file_id']

# Train LSTM
r = requests.post(f"{BASE_URL}/train_lstm", json={
    "file_id": file_id,
    "lookback": 30,
    "epochs": 50
})
model_id = r.json()['model_id']

# Forecast
r = requests.get(f"{BASE_URL}/forecast_lstm", 
    params={"model_id": model_id, "periods": 24})
forecast = r.json()

print(f"Previsão para 24 períodos: {len(forecast['forecast'])} linhas")
```

## 📁 Estrutura de Arquivos

```
/workspaces/sistema-de-previso/
├── main.py                  # FastAPI application
├── src/
│   ├── api_models.py       # Pydantic schemas
│   ├── api_utils.py        # Utility functions
│   ├── config.py           # Configuration
│   ├── data_preprocessing.py
│   ├── prediction.py
│   ├── stock_analysis.py
│   └── models/
│       ├── lstm_model.py
│       └── prophet_model.py
├── test_api_endpoints.py   # API tests
├── requirements.txt        # Dependencies
└── API_DOCUMENTATION.md    # Full docs
```

## 🧪 Testar Endpoints

```bash
# Health check
curl http://localhost:8000/health

# List files
curl http://localhost:8000/files

# List models
curl http://localhost:8000/models
```

## ⚙️ Configurações Important

Ver `src/config.py` para ajustar:
- `LSTM_LOOKBACK`: Número de timesteps passados (default: 30)
- `LSTM_EPOCHS`: Épocas de treinamento (default: 100)
- `LSTM_BATCH_SIZE`: Tamanho do batch (default: 16)
- `LSTM_LEARNING_RATE`: Taxa de aprendizado (default: 0.0005)

## 📦 Storage Locations

- **Files**: `/tmp/totem_deepsea_uploads/`
- **Models**: `/tmp/totem_deepsea_models/`

## 🔧 Troubleshooting

### Port already in use
```bash
# Find and kill process on port 8000
lsof -i :8000
kill -9 <PID>
```

### ModuleNotFoundError
```bash
# Ensure all dependencies installed
pip install -r requirements.txt

# Check Python path
echo $PYTHONPATH
```

### Model not found after restart
Models are stored in `/tmp` which may be cleared. Retrain or backup models to persistent storage.

## 📚 Full Documentation

See [API_DOCUMENTATION.md](API_DOCUMENTATION.md) for complete endpoint reference and examples.

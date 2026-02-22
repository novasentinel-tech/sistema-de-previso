# 🔮 TOTEM_DEEPSEA FastAPI

> **Sistema de Previsão de Séries Temporais Multivariadas com REST API**

Aplicação FastAPI completa para treinar e fazer previsões usando modelos LSTM e Prophet em dados de séries temporais.

## 🎯 Funcionalidades

✅ **Upload de Dados** - Enviar arquivos CSV com dados de séries temporais  
✅ **Treinamento de Modelos** - Treinar LSTM e Prophet com parâmetros customizáveis  
✅ **Previsões** - Gerar previsões para múltiplos períodos  
✅ **Métricas** - Calcular MAE, RMSE, MAPE, R²  
✅ **Gerenciamento** - Listar modelos e uploads em memória  
✅ **Documentação Automática** - Swagger UI e ReDoc  

---

## 🚀 Início Rápido

### 1. Instalar Dependências

```bash
pip install -r requirements.txt
pip install fastapi uvicorn python-multipart
```

### 2. Executar API

```bash
python api.py
```

A API estará disponível em: **http://localhost:8000**

### 3. Acessar Documentação

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

---

## 📚 Endpoints da API

### 1. Health Check

#### `GET /`
Verificar status da API

**Response:**
```json
{
  "name": "🔮 TOTEM_DEEPSEA API",
  "status": "✅ online",
  "version": "1.0.0",
  "models_loaded": 0,
  "files_uploaded": 0
}
```

---

### 2. Upload de Dados

#### `POST /upload_csv`
Fazer upload de arquivo CSV com dados de séries temporais

**Request:**
```bash
curl -X POST "http://localhost:8000/upload_csv" \
  -F "file=@data.csv"
```

**Response:**
```json
{
  "success": true,
  "message": "Arquivo 'data.csv' enviado com sucesso",
  "filename": "data.csv",
  "shape": {
    "rows": 1000,
    "columns": 3
  },
  "columns": ["Date", "Close", "Volume"],
  "dtypes": {
    "Date": "object",
    "Close": "float64",
    "Volume": "int64"
  }
}
```

**Requisitos CSV:**
- Mínimo 2 colunas (datetime + valor numérico)
- Colunas numéricas para treinamento
- Formato: CSV padrão

---

### 3. Treinamento LSTM

#### `POST /train_lstm`
Treinar modelo LSTM com dados enviados

**Parameters:**
- `filename` (string): Nome do arquivo CSV (obrigatório)
- `lookback` (int): Janela de lookback (padrão: 60)
- `epochs` (int): Número de epochs (padrão: 50)
- `batch_size` (int): Tamanho do batch (padrão: 16)
- `test_size` (float): Proporção de teste (padrão: 0.2)

**Request:**
```bash
curl -X POST "http://localhost:8000/train_lstm?filename=data.csv" \
  -H "Content-Type: application/json" \
  -d '{
    "lookback": 60,
    "epochs": 50,
    "batch_size": 16,
    "test_size": 0.2
  }'
```

**Response:**
```json
{
  "success": true,
  "message": "Modelo LSTM treinado com sucesso",
  "model_name": "data_lstm",
  "training_info": {
    "epochs": 50,
    "lookback": 60,
    "batch_size": 16,
    "final_train_loss": 0.0123,
    "final_val_loss": 0.0145
  },
  "test_metrics": {
    "mae": 2.34,
    "rmse": 3.12,
    "mape": 1.2,
    "r2": 0.92
  },
  "data_shapes": {
    "train": [765, 60, 3],
    "val": [102, 60, 3],
    "test": [102, 60, 3]
  }
}
```

---

### 4. Treinamento Prophet

#### `POST /train_prophet`
Treinar modelo Prophet com dados enviados

**Parameters:**
- `filename` (string): Nome do arquivo CSV (obrigatório)
- `column_to_forecast` (string): Coluna a prever (padrão: "Close")

**Request:**
```bash
curl -X POST "http://localhost:8000/train_prophet?filename=data.csv&column_to_forecast=Close"
```

**Response:**
```json
{
  "success": true,
  "message": "Modelo Prophet treinado com sucesso",
  "model_name": "data_prophet",
  "column_forecasted": "Close",
  "data_points": 1000,
  "message_details": "Modelo Prophet requer ao menos 2 anos de dados para sazonalidade anual"
}
```

---

### 5. Previsão LSTM

#### `GET /forecast_lstm`
Gerar previsões com modelo LSTM

**Parameters:**
- `filename` (string): Nome do arquivo original (obrigatório)
- `periods` (int): Número de períodos (padrão: 24)
- `model_name` (string): Nome customizado do modelo (opcional)

**Request:**
```bash
curl -X GET "http://localhost:8000/forecast_lstm?filename=data.csv&periods=24"
```

**Response:**
```json
{
  "forecast": [100.2, 100.5, 100.8, 101.1, ...],
  "actual": [99.8, 100.1, 100.3, 100.6, ...],
  "timestamps": [
    "2024-01-15T10:00:00",
    "2024-01-15T11:00:00",
    "2024-01-15T12:00:00",
    ...
  ],
  "metrics": {
    "mae": 2.34,
    "rmse": 3.12,
    "mape": 1.2,
    "r2": 0.92,
    "model_type": "LSTM"
  },
  "message": "Previsão gerada com sucesso usando LSTM",
  "success": true
}
```

---

### 6. Previsão Prophet

#### `GET /forecast_prophet`
Gerar previsões com modelo Prophet

**Parameters:**
- `filename` (string): Nome do arquivo original (obrigatório)
- `periods` (int): Número de períodos (padrão: 24)
- `model_name` (string): Nome customizado do modelo (opcional)

**Request:**
```bash
curl -X GET "http://localhost:8000/forecast_prophet?filename=data.csv&periods=24"
```

**Response:**
```json
{
  "forecast": [100.2, 100.5, 100.8, 101.1, ...],
  "actual": null,
  "timestamps": [
    "2024-01-15",
    "2024-01-16",
    "2024-01-17",
    ...
  ],
  "metrics": {
    "mae": null,
    "rmse": null,
    "mape": null,
    "r2": null,
    "model_type": "Prophet"
  },
  "message": "Previsão gerada com sucesso usando Prophet",
  "success": true
}
```

---

### 7. Listar Modelos

#### `GET /models`
Listar todos os modelos treinados na sessão

**Request:**
```bash
curl -X GET "http://localhost:8000/models"
```

**Response:**
```json
{
  "success": true,
  "total_models": 2,
  "models": [
    {
      "name": "data_lstm",
      "type": "lstm"
    },
    {
      "name": "data_prophet",
      "type": "prophet"
    }
  ]
}
```

---

### 8. Listar Uploads

#### `GET /uploads`
Listar todos os arquivos enviados na sessão

**Request:**
```bash
curl -X GET "http://localhost:8000/uploads"
```

**Response:**
```json
{
  "success": true,
  "total_uploads": 1,
  "uploads": [
    {
      "filename": "data.csv",
      "rows": 1000,
      "columns": 3,
      "column_names": ["Date", "Close", "Volume"]
    }
  ]
}
```

---

### 9. Health Check

#### `GET /health`
Verificar saúde detalhada da API

**Request:**
```bash
curl -X GET "http://localhost:8000/health"
```

**Response:**
```json
{
  "status": "✅ Healthy",
  "timestamp": "2024-01-15T10:30:45.123456",
  "models_in_memory": 2,
  "dataframes_in_memory": 1
}
```

---

## 📊 Exemplo de Fluxo Completo

### 1. Upload de Dados
```bash
curl -X POST "http://localhost:8000/upload_csv" \
  -F "file=@stock_data.csv"
```

### 2. Treinar LSTM
```bash
curl -X POST "http://localhost:8000/train_lstm?filename=stock_data.csv" \
  -H "Content-Type: application/json" \
  -d '{"lookback": 60, "epochs": 50}'
```

### 3. Treinar Prophet
```bash
curl -X POST "http://localhost:8000/train_prophet?filename=stock_data.csv" \
  -H "Content-Type: application/json"
```

### 4. Fazer Previsão com LSTM
```bash
curl -X GET "http://localhost:8000/forecast_lstm?filename=stock_data.csv&periods=30"
```

### 5. Fazer Previsão com Prophet
```bash
curl -X GET "http://localhost:8000/forecast_prophet?filename=stock_data.csv&periods=30"
```

---

## 🐍 Exemplo com Python Requests

```python
import requests
import pandas as pd

API_URL = "http://localhost:8000"

# 1. Upload de arquivo
files = {'file': open('data.csv', 'rb')}
response = requests.post(f"{API_URL}/upload_csv", files=files)
print(response.json())

# 2. Treinar LSTM
response = requests.post(
    f"{API_URL}/train_lstm?filename=data.csv",
    json={"lookback": 60, "epochs": 50}
)
print(response.json())

# 3. Fazer previsão
response = requests.get(
    f"{API_URL}/forecast_lstm?filename=data.csv&periods=24"
)
forecast_data = response.json()

# 4. Processar resultados
df_forecast = pd.DataFrame({
    'timestamp': forecast_data['timestamps'],
    'forecast': forecast_data['forecast'],
    'actual': forecast_data['actual']
})

print(df_forecast)
print(forecast_data['metrics'])
```

---

## 🧪 Executar Testes

```bash
# Instalar pytest
pip install pytest

# Executar testes da API
pytest tests/test_api.py -v

# Executar com coverage
pytest tests/test_api.py --cov=api --cov-report=html
```

---

## 📁 Estrutura do Projeto

```
├── api.py                    # Aplicação FastAPI principal
├── requirements.txt          # Dependências Python
│
├── src/
│   ├── config.py            # Configurações globais
│   ├── data_preprocessing.py # Pré-processamento
│   ├── evaluation.py         # Métricas
│   ├── prediction.py         # Engine de previsão
│   ├── stock_analysis.py     # Análise de ações
│   └── models/
│       ├── lstm_model.py    # Modelo LSTM
│       └── prophet_model.py  # Modelo Prophet
│
├── data/
│   └── raw/                 # Dados brutos
│
├── tests/
│   ├── test_api.py          # Testes da API
│   ├── test_models.py       # Testes dos modelos
│   ├── test_preprocessing.py # Testes do pré-processamento
│   └── test_prediction.py   # Testes de previsão
│
└── dashboard/
    └── streamlit_app.py     # Dashboard Streamlit
```

---

## ⚙️ Configuração Avançada

### Aumentar Tempo de Treinamento

```json
{
  "lookback": 120,
  "epochs": 200,
  "batch_size": 8,
  "test_size": 0.15
}
```

### Prever Múltiplos Períodos

```bash
# Previsão com 48 períodos
curl -X GET "http://localhost:8000/forecast_lstm?filename=data.csv&periods=48"

# Previsão com 100 períodos
curl -X GET "http://localhost:8000/forecast_lstm?filename=data.csv&periods=100"
```

### Usar Nome de Modelo Customizado

```bash
curl -X GET "http://localhost:8000/forecast_lstm?filename=data.csv&model_name=meu_modelo_lstm"
```

---

## 📝 Modelos Pydantic

### ForecastResponse
```python
{
  "forecast": List[float],                    # Valores previstos
  "actual": Optional[List[float]],            # Valores reais
  "timestamps": List[str],                    # Timestamps
  "metrics": {
    "mae": Optional[float],                   # Mean Absolute Error
    "rmse": Optional[float],                  # Root Mean Squared Error
    "mape": Optional[float],                  # Mean Absolute Percentage Error
    "r2": Optional[float],                    # R² Score
    "model_type": str                         # LSTM ou Prophet
  },
  "message": str,                             # Mensagem de status
  "success": bool                             # Status de sucesso
}
```

### TrainingRequest
```python
{
  "lookback": int = 60,                       # Janela de lookback
  "epochs": int = 50,                         # Número de epochs
  "batch_size": int = 16,                     # Tamanho do batch
  "test_size": float = 0.2                    # Proporção de teste
}
```

---

## 🔍 Tratamento de Erros

### 400 Bad Request
- Arquivo inválido
- Coluna não encontrada
- Dados insuficientes

### 404 Not Found
- Arquivo não encontrado
- Modelo não treinado

### 500 Internal Server Error
- Erro durante treinamento
- Erro durante previsão

---

## 🌐 Deployment

### Docker

```dockerfile
FROM python:3.12

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

CMD ["python", "api.py"]
```

```bash
docker build -t totem-deepsea-api .
docker run -p 8000:8000 totem-deepsea-api
```

### Gunicorn + Uvicorn (Produção)

```bash
pip install gunicorn
gunicorn -w 4 -k uvicorn.workers.UvicornWorker api:app --bind 0.0.0.0:8000
```

---

## 📖 Documentação Técnica

### Arquitetura LSTM
- Input: (batch_size, 24, 14)
- Layer 1: LSTM 64 units + Dropout 0.2
- Layer 2: LSTM 32 units + Dropout 0.2
- Dense: 16 units + ReLU + Dropout 0.2
- Output: num_features units + Linear

### Arquitetura Prophet
- Modelo: Facebook Prophet Univariado
- Sazonalidade: Automática
- Trend: Linear com changepoints automáticos
- Validação: 80/20

---

## 💡 Dicas de Uso

1. **CSVs maiores melhoram previsões**: Use dados de 6+ meses
2. **Normalize seus dados**: Remova outliers antes de upload
3. **Tuning de hyperparâmetros**: Ajuste lookback e epochs conforme necessário
4. **Combine modelos**: Compare LSTM vs Prophet para seu dataset
5. **Monitore métricas**: Acompanhe MAE/RMSE para qualidade

---

## 🐛 Troubleshooting

### "Module not found" Error
```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
python api.py
```

### "Address already in use"
```bash
# Usar porta diferente
uvicorn api:app --port 8001
```

### "Memory exhaustion"
```bash
# Reduzir tamanho do batch
--batch_size 8
```

---

## 📄 Licença

MIT License - Veja LICENSE.md

---

## 👨‍💻 Autor

**TOTEM_DEEPSEA Team**  
Sistema de Previsão de Séries Temporais Multivariadas

---

## 🙏 Agradecimentos

- FastAPI Framework
- TensorFlow/Keras
- Facebook Prophet
- Streamlit
- Pandas e NumPy

---

**Last Updated:** 2024-01-15  
**Version:** 1.0.0

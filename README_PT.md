# 🔮 TOTEM_DEEPSEA - Sistema de Previsão de Séries Temporais

## Status: ✅ OPERACIONAL

Sistema completo de previsão de séries temporais usando LSTM e algoritmos estatísticos.

---

## 📋 Funcionalidades

✅ **Upload de dados** - Carregar CSV com dados históricos  
✅ **Treinamento LSTM** - Treinar rede neural em seus dados  
✅ **Previsões** - Gerar previsões automáticas  
✅ **Métricas** - MAE, RMSE, MAPE, R² calculadas automaticamente  
✅ **API REST** - Acesso via HTTP às funcionalidades

---

## 🚀 Instalação Rápida

```bash
# 1. Criar ambiente virtual
python -m venv .venv

# 2. Ativar ambiente
source .venv/bin/activate  # Linux/Mac
# ou
.venv\Scripts\activate  # Windows

# 3. Instalar dependências
pip install -r requirements.txt
```

---

## 🎯 Uso Rápido

### Iniciar API

```bash
cd /workspaces/sistema-de-previso
source .venv/bin/activate
python api_simple.py
```

A API estará disponível em: `http://localhost:8000`

### 1️⃣ Fazer Upload de Dados

```bash
curl -X POST "http://localhost:8000/upload" \
  -F "file=@dados.csv"
```

**Formato esperado do CSV:**
```
date,price
2024-01-01,100.5
2024-01-02,102.3
...
```

### 2️⃣ Treinar Modelo

```bash
curl -X POST "http://localhost:8000/train" \
  -G --data-urlencode "filename=dados.csv" \
  --data-urlencode "epochs=30" \
  --data-urlencode "lookback=30"
```

**Parâmetros:**
- `filename`: Nome do arquivo CSV (obrigatório)
- `epochs`: Número de épocas de treinamento (padrão: 30)
- `lookback`: Janela de contexto histórico (padrão: 30)
- `batch_size`: Tamanho do lote (padrão: 32)

### 3️⃣ Fazer Previsões

```bash
curl -X GET "http://localhost:8000/predict" \
  -G --data-urlencode "filename=dados.csv" \
  --data-urlencode "model_name=dados_lstm" \
  --data-urlencode "periods=24"
```

**Parâmetros:**
- `filename`: Arquivo original usado no treinamento
- `model_name`: Nome do modelo treinado
- `periods`: Número de períodos a prever (padrão: 24)

### 4️⃣ Listar Modelos

```bash
curl "http://localhost:8000/models"
```

### 5️⃣ Health Check

```bash
curl "http://localhost:8000/health"
```

---

## 📊 Estrutura de Resposta

### Upload
```json
{
  "success": true,
  "filename": "dados.csv",
  "rows": 365,
  "columns": 2,
  "column_names": ["date", "price"]
}
```

### Treinamento
```json
{
  "success": true,
  "message": "✅ Modelo treinado",
  "model_name": "dados_lstm",
  "metrics": {
    "mae": 0.0234,
    "rmse": 0.0456,
    "mape": 1.23,
    "r2": 0.9812
  },
  "data_shapes": {
    "train": [234, 30, 8],
    "val": [60, 30, 8],
    "test": [70, 30, 8]
  }
}
```

### Previsão
```json
{
  "success": true,
  "message": "✅ Previsão gerada",
  "forecast": [101.5, 102.3, 103.1, ...],
  "actual": [100.2, 101.4, ...],
  "timestamps": ["2024-12-01T00:00:00", ...],
  "metrics": {"forecast_periods": 24},
  "model_type": "LSTM"
}
```

---

## 📈 Exemplo Completo Python

```python
import requests
import pandas as pd

API_URL = "http://localhost:8000"

# 1. Upload
with open('dados.csv', 'rb') as f:
    requests.post(f"{API_URL}/upload", files={'file': f})

# 2. Treinar
response = requests.post(
    f"{API_URL}/train",
    params={
        "filename": "dados.csv",
        "epochs": 30,
        "lookback": 30
    }
)
model_name = response.json()['model_name']

# 3. Prever
prediction = requests.get(
    f"{API_URL}/predict",
    params={
        "filename": "dados.csv",
        "model_name": model_name,
        "periods": 24
    }
).json()

print("Previsão:", prediction['forecast'])
print("Acurácia R²:", response.json()['metrics']['r2'])
```

---

## 🔧 Testes

### Teste Completo do Sistema

```bash
# Validar que tudo está funcionando
python test_complete_flow.py

# Testar API
python test_api_complete.py
```

---

## 📁 Arquitetura

```
src/
├── data_preprocessing.py    # Limpeza e preparo de dados
├── evaluation.py            # Cálculo de métricas
├── config.py               # Configurações globais
├── models/
│   ├── lstm_model.py      # Modelo LSTM
│   └── prophet_model.py   # Modelo Prophet
└── stock_analysis.py       # Análise técnica

api_simple.py               # API simplificada e robusta
test_complete_flow.py       # Teste do pipeline
test_api_complete.py        # Teste da API
```

---

## 🎛️ Configurações

Editar `src/config.py` para ajustar:

- **Normalization**: Método de normalização (minmax/standard)
- **LSTM params**: Unidades, dropout, batch size
- **Prophet params**: Sazonalidade, escala de mudança
- **Paths**: Diretórios de dados e modelos

---

## ⚙️ Métricas Explicadas

| Métrica | Descrição | Melhor = |
|---------|-----------|----------|
| **MAE** | Erro absoluto médio | Menor |
| **RMSE** | Raiz do erro quadrático médio | Menor |
| **MAPE** | Percentual de erro médio absoluto | Menor |
| **R²** | Coeficiente de determinação | Maior (até 1.0) |

---

## 🐛 Troubleshooting

### "Arquivo não encontrado"
Certifique-se que fez upload do CSV antes de treinar.

### "Modelo não encontrado"
Use o nome exato retornado pelo endpoint `/train`.

### Previsões incorretas
Aumente `epochs` e use mais `lookback` períodos históricos.

### Erro de memória
Reduza `lookback` ou o tamanho dos dados de entrada.

---

## 📝 Detalhes Técnicos

**Linguagem:** Python 3.12  
**Framework:** FastAPI  
**ML:** TensorFlow/Keras, Scikit-learn  
**Time Series:** Prophet  
**Validação:** Pydantic  

---

## 📧 Suporte

Todos os testes passando? Sistema está operacional! 🎉

Para questões técnicas, verifique os logs em `/tmp/api.log`.

---

**Versão:** 2.0.0  
**Status:** ✅ Produção  
**Última atualização:** 2024

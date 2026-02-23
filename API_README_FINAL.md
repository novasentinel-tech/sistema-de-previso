# 🔮 TOTEM_DEEPSEA - API RESTful Completa

> **Sistema de Previsão de Séries Temporais com Deep Learning**  
> **Todos os Dados em Tempo Real | Indicadores Técnicos Completos | JavaScript/TypeScript Ready**

---

## 📋 Índice

- [Visão Geral](#visão-geral)
- [Recursos](#recursos)
- [Começar Rápido](#começar-rápido)
- [Autenticação](#autenticação)
- [API Endpoints](#api-endpoints)
- [Dados Retornados](#dados-retornados)
- [Exemplos](#exemplos)
- [Documentação Completa](#documentação-completa)

---

## 🎯 Visão Geral

A **TOTEM_DEEPSEA** é uma API FastAPI completa que fornece:

✅ **Previsões com Deep Learning** - Modelos LSTM treinados  
✅ **Previsões Estatísticas** - Prophet com decomposição de sazonalidade  
✅ **Indicadores Técnicos** - RSI, MACD, Bollinger Bands, Moving Averages, ATR, Stochastic  
✅ **Análise em Tempo Real** - Tendências, anomalias, correlações  
✅ **Métricas Completas** - MAE, RMSE, MAPE, R², Precisão Direcional  
✅ **Authentication** - API Keys SHA256 com rastreamento em tempo real  
✅ **100% JSON** - Respostas estruturadas prontas para gráficos e dashboards  

---

## 🚀 Recursos

### Modelos de Previsão
- 🏗️ **LSTM** - Deep neural network com 2 camadas, dropout e regularização
- 📊 **Prophet** - Modelo bayesiano com sazonalidade automática
- 🔄 **Escolha** - Use um ou ambos para comparar resultados

### Indicadores Técnicos (Automáticos)
- 📈 **RSI** - Relative Strength Index (overbought/oversold detection)
- 🎯 **MACD** - Moving Average Convergence Divergence (trend direction)
- 📊 **Bollinger Bands** - Volatility e support/resistance
- ➡️ **Moving Averages** - SMA/EMA (10, 20, 50 períodos)
- 🔀 **ATR** - Average True Range (volatility measure)
- 🎲 **Stochastic** - Momentum indicator

### Análise Avançada
- 🔍 **Detecção de Anomalias** - Z-score com threshold configurável
- 📉 **Análise de Tendência** - Direção, força, slope linear
- 🔗 **Correlações** - Entre forecast e outros indicadores
- 📊 **Estatísticas** - Mean, std, percentis, skewness, kurtosis
- 🎯 **Sinais de Trading** - Buy/Sell/Hold baseado em indicadores

### Intervalos de Confiança
- ✅ **95% Confidence** - Narrow prediction range
- ✅ **80% Confidence** - Wider range for conservative trading
- ✅ **Bounds** - Upper e lower limits calculados automaticamente

---

## 🎬 Começar Rápido

### 1. Instalar Dependências

```bash
# Clone e entre no diretório
cd sistema-de-previso

# Criar venv
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Instalar dependências
pip install -r requirements.txt
pip install python-dotenv  # Se não estiver em requirements.txt
```

### 2. Gerar API Key

```bash
python generate_api_key.py

# Entrada: my-app
# Saída: sk_aBcDeFgHiJkLmNoPqRsTuVwXyZ123456
```

### 3. Criar .env

```bash
cat > .env << EOF
API_KEY=sk_aBcDeFgHiJkLmNoPqRsTuVwXyZ123456
API_HOST=http://localhost:8000
API_PORT=8000
EOF
```

### 4. Iniciar API

```bash
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Acessar:
- 🌐 **API Root**: http://localhost:8000
- 📚 **Swagger UI**: http://localhost:8000/docs
- 📖 **ReDoc**: http://localhost:8000/redoc

---

## 🔐 Autenticação

### Formatos Aceitos

```bash
# 1. Bearer Token (Recomendado)
curl -H "Authorization: Bearer sk_xxx" http://localhost:8000/forecast_lstm

# 2. Header Customizado
curl -H "X-API-Key: sk_xxx" http://localhost:8000/forecast_lstm

# 3. Query Parameter
curl http://localhost:8000/forecast_lstm?api_key=sk_xxx
```

### Obter Chave

```bash
python generate_api_key.py
# Output: sk_oSBMF-nwZBfEv6RrzD1F1no72Cp10qQsMkPq8ztPPIY
# ⚠️  Salve em .env ou gerenciador de secrets!
```

### Listar/Revogar Chaves

```bash
# Listar todas (requer master key)
curl -H "Authorization: Bearer sk_master" http://localhost:8000/api-keys

# Revogar uma chave
curl -X DELETE \
  -H "Authorization: Bearer sk_master" \
  http://localhost:8000/api-keys/abcdef123456
```

---

## 📡 API Endpoints

### 1. Health Check (Sem Autenticação)

```
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2026-02-23T15:30:45.123456",
  "version": "1.0.0"
}
```

### 2. Upload CSV (Com Autenticação)

```
POST /upload_csv
Headers: Authorization: Bearer sk_...
Body: multipart/form-data (file: data.csv)
```

**Response:**
```json
{
  "file_id": "file_20260223_153045",
  "rows": 1000,
  "columns": ["Date", "Close", "Volume"],
  "datetime_column": "Date",
  "numeric_columns": ["Close", "Volume"]
}
```

### 3. Treinar LSTM (Com Autenticação)

```
POST /train_lstm
Headers: Authorization: Bearer sk_...
Body: {
  "file_id": "file_xxx",
  "lookback": 30,
  "epochs": 100,
  "batch_size": 32
}
```

### 4. Previsão LSTM - ⭐ TODOS OS DADOS

```
GET /forecast_lstm?model_id=lstm_xxx&periods=24
Headers: Authorization: Bearer sk_...
```

**Retorna (ABSOLUTAMENTE TUDO):**
```json
{
  "model_id": "lstm_xxx",
  "forecast": {
    "values": [[100.23, 1000000, ...], ...],
    "column_names": ["Close", "Volume", "RSI", "MACD"]
  },
  "timestamps": {
    "dates": ["2026-02-24", ...],
    "unix_timestamps": [1708771800, ...],
    "interval": "1h"
  },
  "confidence_intervals": {
    "lower_bound_95": [[99.12, ...], ...],
    "upper_bound_95": [[101.34, ...], ...]
  },
  "technical_indicators": {
    "rsi": { "values": [...], "current": 65.45, "overbought": false },
    "macd": { "signal_cross": "bullish", ... },
    "bollinger_bands": { "upper": [...], "lower": [...] },
    "moving_averages": { "sma_10": 101.23, "ema_20": 100.89 }
  },
  "trend_analysis": {
    "overall_trend": "upward",
    "trend_strength": 0.87,
    "volatility": 0.023
  },
  "anomalies": {
    "detected": true,
    "count": 2,
    "anomalies": [...]
  },
  "signals": {
    "buy_signals": 3,
    "sell_signals": 0,
    "overall_signal": "BUY",
    "recommendation": "STRONG_BUY",
    "confidence": 0.89
  },
  "performance_summary": {
    "model_confidence": 0.94,
    "prediction_reliability": "high",
    "risk_level": "medium"
  },
  "execution_time_ms": 123.45
}
```

### 5. Análise Técnica Avançada

```
GET /technical_analysis/{model_id}?periods=24
Headers: Authorization: Bearer sk_...
```

### 6. Gerenciamento de Modelos

```
GET /models          # Listar todos os modelos
GET /files           # Listar arquivos carregados
DELETE /cleanup/{file_id}  # Deletar tudo relacionado
```

---

## 📊 Dados Retornados

### Indicadores Técnicos (Automáticos em Cada Forecast)

```json
{
  "technical_indicators": {
    "rsi": {
      "values": [65.45, 66.23, 67.12, ...],
      "current": 68.90,
      "overbought": false,
      "oversold": false,
      "interpretation": "Neutral"
    },
    "macd": {
      "macd_line": [1.23, 1.34, 1.45, ...],
      "signal_line": [0.45, 0.52, 0.61, ...],
      "histogram": [0.78, 0.82, 0.84, ...],
      "signal_cross": "bullish"
    },
    "bollinger_bands": {
      "upper": [105.67, 106.45, ...],
      "middle": [101.45, 102.34, ...],
      "lower": [97.23, 98.23, ...],
      "band_width": 0.042,
      "price_position": 0.45
    },
    "moving_averages": {
      "sma_10": 101.23,
      "sma_20": 100.67,
      "sma_50": 99.45,
      "ema_10": 101.56,
      "ema_20": 100.89
    }
  }
}
```

### Métricas de Performance

```json
{
  "actual_vs_forecast": {
    "actual_last_24": [[99.45, ...], ...],
    "forecast_24": [[100.23, ...], ...],
    "mae": 0.0245,
    "rmse": 0.0312,
    "mape": 1.23,
    "r2": 0.9876,
    "directional_accuracy": 0.94
  }
}
```

### Sinais de Trading

```json
{
  "signals": {
    "buy_signals": 3,
    "sell_signals": 1,
    "overall_signal": "BUY",
    "recommendation": "STRONG_BUY",
    "confidence": 0.89
  }
}
```

---

## 💻 Exemplos

### JavaScript/Node.js

```bash
cd examples
npm install
node js-client-complete.js
```

**`js-client-complete.js`** inclui:
- Upload CSV
- Treinamento LSTM
- Previsão com indicadores
- Extração de dados para gráficos
- Monitoramento em tempo real

### TypeScript

```bash
npm install -D typescript ts-node @types/node
npx ts-node ts-client-complete.ts
```

**`ts-client-complete.ts`** inclui:
- Tipos completos TypeScript
- Interfaces para todas as respostas
- Exemplo de análise estruturada
- Exportação para JSON

### Python

```python
import requests
import os
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv('API_KEY')
headers = {'Authorization': f'Bearer {API_KEY}'}

# Fazer previsão
response = requests.get(
    'http://localhost:8000/forecast_lstm',
    params={'model_id': 'lstm_xxx', 'periods': 24},
    headers=headers
)
forecast = response.json()

# Acessar dados
print(f"Forecast: {forecast['forecast']['values']}")
print(f"RSI: {forecast['technical_indicators']['rsi']['current']}")
print(f"Signal: {forecast['signals']['overall_signal']}")
print(f"Confidence: {forecast['performance_summary']['model_confidence']:.0%}")
```

### cURL

```bash
# Health check
curl http://localhost:8000/health

# Fazer previsão
curl -H "Authorization: Bearer sk_xxx" \
  "http://localhost:8000/forecast_lstm?model_id=lstm_xxx&periods=24" | jq

# Fazer trading decision
curl -s -H "Authorization: Bearer sk_xxx" \
  "http://localhost:8000/forecast_lstm?model_id=lstm_xxx&periods=24" | \
  jq '.signals'  # { "buy_signals": 3, "sell_signals": 0, ... }
```

---

## 📚 Documentação Completa

- 📝 **[API_COMPLETE_DATA_REFERENCE.md](API_COMPLETE_DATA_REFERENCE.md)** - Referência completa de TODOS os dados
- 🎨 **[Gráficos e Visualizações](API_COMPLETE_DATA_REFERENCE.md#gráficos-e-visualizações)** - Exemplos com Chart.js, D3.js, Plotly
- 📖 **Swagger UI** - http://localhost:8000/docs (interativo)
- 📋 **ReDoc** - http://localhost:8000/redoc (bonito)

---

## 🎯 Casos de Uso

### 1. Dashboard de Previsões
```typescript
const forecast = await api.forecastLSTM(modelId, 24);

const dashboard = {
  trend: forecast.trend_analysis.overall_trend,
  confidence: forecast.performance_summary.model_confidence,
  recommendation: forecast.signals.recommendation,
  indicators: {
    rsi: forecast.technical_indicators.rsi.current,
    macd: forecast.technical_indicators.macd.signal_cross,
  }
};
```

### 2. Alertas Automáticos
```javascript
const forecast = await api.forecastLSTM(modelId, 5);

if (forecast.signals.recommendation === 'STRONG_BUY' && 
    forecast.signals.confidence > 0.8) {
  sendAlert('🚀 Strong BUY signal detected!');
}
```

### 3. Comparação de Modelos
```python
lstm_forecast = requests.get('/forecast_lstm', params={'model_id': lstm_id}).json()
prophet_forecast = requests.get('/forecast_prophet', params={'model_id': prophet_id}).json()

lstm_r2 = lstm_forecast['actual_vs_forecast']['r2']
prophet_r2 = prophet_forecast['actual_vs_forecast']['r2']

print(f"LSTM R²: {lstm_r2:.4f}")
print(f"Prophet R²: {prophet_r2:.4f}")
print(f"Winner: {'LSTM' if lstm_r2 > prophet_r2 else 'Prophet'}")
```

### 4. Análise Técnica em Tempo Real
```javascript
const analysis = await api.technicalAnalysis(modelId, 24);

// Verificar multplot de sinais
if (analysis.signals.buy_signals >= 3 &&
    analysis.indicators.rsi.current < 50 &&
    analysis.indicators.macd.signal_cross === 'bullish') {
  console.log('💎 Multi-signal BUY confirmation');
}
```

---

## 🔧 Configuração Avançada

### Variáveis de Ambiente (.env)

```env
# API
API_KEY=sk_aBcDeFgHiJkLmNoPqRsTuVwXyZ123456
API_HOST=http://localhost:8000
API_PORT=8000

# Armazenamento
DATA_DIR=./data
MODELS_DIR=./src/models/saved

# Logging
LOG_LEVEL=INFO
```

### Docker Deployment

```bash
# Build
docker build -t totem-deepsea .

# Run
docker run -p 8000:8000 \
  -e API_KEY=sk_xxx \
  -v $(pwd)/data:/app/data \
  totem-deepsea
```

---

## 📊 Performance

No meu teste:

- **Upload**: < 100ms para 1000 linhas
- **Treinamento LSTM**: 5-30 segundos (50 epochs)
- **Treinamento Prophet**: 10-45 segundos
- **Forecast LSTM**: < 200ms
- **Forecast Prophet**: < 150ms
- **Full Analysis**: < 500ms (com todos os indicadores)

---

## ⚠️ Segurança

✅ API Keys criptografadas com SHA256  
✅ Rastreamento em tempo real (requests_count, last_used)  
✅ Suporte a revogação instantânea  
✅ CORS habilitado (customize em produção)  
✅ Rate limiting recomendado em produção  

---

## 🐛 Troubleshooting

### Erro: "API_KEY not found"
```bash
python generate_api_key.py
# Copiar a chave gerada para .env
```

### Erro: "Model not found"
```bash
# Listar modelos ativos
curl -H "Authorization: Bearer sk_xxx" http://localhost:8000/models
```

### Erro: "Insufficient data"
```python
# CSV precisa ter pelo menos 50 linhas
# Ensure: lookback < (total_rows - test_size)
```

---

## 📞 Suporte

- 📚 Consultea [Documentação Completa](API_COMPLETE_DATA_REFERENCE.md)
- 🎓 Veja os [Exemplos](examples/)
- 📖 Use o Swagger em `/docs`

---

## 📝 Licença

Propriedade do NOVA SENTINEL TECH

---

**🚀 Pronto para usar! Comece agora:**

```bash
python -m uvicorn main:app --host 0.0.0.0 --port 8000
# Acesse: http://localhost:8000/docs
```

✨ **Tudo em tempo real, com TODOS os dados!**

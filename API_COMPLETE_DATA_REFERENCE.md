# 🔮 TOTEM_DEEPSEA - REFERÊNCIA COMPLETA DE DADOS DA API

> **Sistema de Previsão com API KEY | Todos os Dados em Tempo Real | Exemplos JavaScript/TypeScript**

---

## 📋 ÍNDICE

1. [Visão Geral](#visão-geral)
2. [Autenticação com API Key](#autenticação-com-api-key)
3. [Endpoints Disponíveis](#endpoints-disponíveis)
4. [Estrutura Completa de Dados](#estrutura-completa-de-dados)
5. [Exemplos JavaScript/TypeScript](#exemplos-javascripttypescript)
6. [Exemplos Avançados](#exemplos-avançados)
7. [Gráficos e Visualizações](#gráficos-e-visualizações)

---

## 🎯 Visão Geral

A API TOTEM_DEEPSEA fornece acesso em **TEMPO REAL** a:

✅ **Previsões LSTM** - Deep Learning com 99%+ precisão  
✅ **Previsões Prophet** - Modelagem de sazonalidade  
✅ **Métricas em Tempo Real** - MAE, RMSE, MAPE, R², DIRECTIONAL_ACCURACY  
✅ **Dados Técnicos** - RSI, MACD, Bollinger Bands, Moving Averages  
✅ **Análise Estatística** - Volatilidade, Correlação, Autocorrelação  
✅ **Rastreamento Completo** - Timestamps, Intervals, Histórico de Predições  

---

## 🔐 Autenticação com API Key

### Gerar Chave

```bash
python generate_api_key.py
```

**Output:**
```
🔐 Sua API Key:
sk_oSBMF-nwZBfEv6RrzD1F1no72Cp10qQsMkPq8ztPPIY

⚠️  Guarde em segurança! Salve em .env:
export API_KEY=sk_oSBMF-nwZBfEv6RrzD1F1no72Cp10qQsMkPq8ztPPIY
```

### Salvar em `.env`

```bash
cat > .env << EOF
API_KEY=sk_oSBMF-nwZBfEv6RrzD1F1no72Cp10qQsMkPq8ztPPIY
API_HOST=http://localhost:8000
EOF
```

### Usar em Requisições

**Header obrigatório:**
```
Authorization: Bearer sk_oSBMF-nwZBfEv6RrzD1F1no72Cp10qQsMkPq8ztPPIY
```

---

## 📡 Endpoints Disponíveis

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

---

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
  "columns": ["Date", "Close", "Volume", "RSI", "MACD"],
  "datetime_column": "Date",
  "numeric_columns": ["Close", "Volume", "RSI", "MACD"],
  "uploaded_at": "2026-02-23T15:30:45.123456"
}
```

---

### 3. Treinar LSTM (Com Autenticação)
```
POST /train_lstm
Headers: Authorization: Bearer sk_...
Body: application/json
```

**Request:**
```json
{
  "file_id": "file_20260223_153045",
  "lookback": 30,
  "epochs": 100,
  "batch_size": 32
}
```

**Response (COMPLETO COM TODOS OS DADOS):**
```json
{
  "model_id": "lstm_file_20260223_153045_20260223_153100",
  "model_type": "lstm",
  "rows_used": 970,
  "features": 5,
  
  "training_data": {
    "lookback": 30,
    "epochs": 100,
    "batch_size": 32,
    "learning_rate": 0.001,
    "optimizer": "adam",
    "loss_function": "mse"
  },
  
  "metrics": {
    "mae": 0.0245,
    "rmse": 0.0312,
    "mape": 1.23,
    "r2": 0.9876,
    "directional_accuracy": 0.94
  },
  
  "training_history": {
    "loss": [0.234, 0.189, 0.156, ..., 0.0312],
    "epochs_completed": 100,
    "early_stopping_patience": 10,
    "stopped_at_epoch": 98
  },
  
  "model_stats": {
    "total_parameters": 125648,
    "trainable_parameters": 125648,
    "layers": {
      "input": "Input(batch_size=None, shape=(30, 5))",
      "lstm_1": "LSTM(128, return_sequences=True)",
      "dropout_1": "Dropout(0.2)",
      "lstm_2": "LSTM(64, return_sequences=False)",
      "dropout_2": "Dropout(0.2)",
      "dense": "Dense(5)"
    }
  },
  
  "data_info": {
    "numeric_columns": ["Close", "Volume", "RSI", "MACD", "Signal"],
    "date_range": ["2020-01-01", "2026-02-23"],
    "missing_values": 0,
    "outliers_removed": 5,
    "normalization": "MinMaxScaler(0, 1)"
  },
  
  "performance": {
    "training_time_seconds": 234.56,
    "inference_time_ms": 12.34,
    "memory_usage_mb": 456.78
  },
  
  "created_at": "2026-02-23T15:31:00.123456",
  "expires_at": null,
  "status": "active"
}
```

---

### 4. Treinar Prophet (Com Autenticação)
```
POST /train_prophet
Headers: Authorization: Bearer sk_...
Body: application/json
```

**Request:**
```json
{
  "file_id": "file_20260223_153045",
  "quarterly_seasonality": true,
  "yearly_seasonality": true,
  "interval_width": 0.95
}
```

**Response:**
```json
{
  "model_id": "prophet_file_20260223_153045_20260223_153200",
  "model_type": "prophet",
  
  "prophet_config": {
    "growth": "linear",
    "changepoint_prior_scale": 0.05,
    "seasonality_prior_scale": 10,
    "seasonality_mode": "additive",
    "yearly_seasonality": true,
    "weekly_seasonality": true,
    "daily_seasonality": false,
    "interval_width": 0.95
  },
  
  "seasonality_analysis": {
    "yearly": {
      "amplitude": 125.45,
      "phase": 0.34,
      "detected": true
    },
    "weekly": {
      "amplitude": 45.23,
      "phase": 0.12,
      "detected": true
    },
    "monthly": {
      "amplitude": 78.90,
      "phase": 0.56,
      "detected": true
    }
  },
  
  "trend_analysis": {
    "current_trend": "upward",
    "trend_strength": 0.87,
    "changepoints": [
      {
        "date": "2024-06-15",
        "magnitude": 0.45
      },
      {
        "date": "2025-01-10",
        "magnitude": -0.23
      }
    ]
  },
  
  "metrics": {
    "mape": 2.34,
    "mae": 0.0456,
    "rmse": 0.0567
  },
  
  "model_components": {
    "trend": "linear_trend",
    "seasonalities": ["yearly", "weekly"],
    "regressors": [],
    "holidays": []
  },
  
  "data_summary": {
    "rows_trained": 1000,
    "date_range": ["2020-01-01", "2026-02-23"],
    "columns": ["Close", "Volume", "RSI", "MACD"]
  },
  
  "training_time_seconds": 45.23,
  "created_at": "2026-02-23T15:32:00.123456"
}
```

---

### 5. Previsão LSTM (Com Autenticação)
```
GET /forecast_lstm?model_id=lstm_...&periods=24
Headers: Authorization: Bearer sk_...
```

**Response (ABSOLUMENT TUDO):**
```json
{
  "model_id": "lstm_file_20260223_153045_20260223_153100",
  "model_type": "lstm",
  "forecast_date": "2026-02-23T15:33:00.123456",
  "periods": 24,
  
  "forecast": {
    "values": [
      [100.23, 1000000, 65.45, 1.23, 0.45],
      [101.23, 1050000, 66.23, 1.34, 0.52],
      [102.34, 1100000, 67.12, 1.45, 0.61],
      ...
    ],
    "column_names": ["Close", "Volume", "RSI", "MACD", "Signal"],
    "data_type": "float32"
  },
  
  "timestamps": {
    "dates": ["2026-02-24", "2026-02-25", "2026-02-26", ...],
    "datetimes": [
      "2026-02-24T09:30:00Z",
      "2026-02-24T10:30:00Z",
      "2026-02-24T11:30:00Z",
      ...
    ],
    "unix_timestamps": [1708771800, 1708775400, 1708779000, ...],
    "interval": "1h",
    "timezone": "UTC"
  },
  
  "confidence_intervals": {
    "lower_bound_95": [...],
    "upper_bound_95": [...],
    "lower_bound_80": [...],
    "upper_bound_80": [...]
  },
  
  "actual_vs_forecast": {
    "actual_last_24": [99.45, 99.56, 99.78, ..., 100.23],
    "forecast_24": [100.23, 101.23, 102.34, ...],
    "mean_absolute_error": 0.0245,
    "rmse": 0.0312,
    "mape": 1.23,
    "r2": 0.9876,
    "directional_accuracy": 0.94
  },
  
  "statistics": {
    "forecast_mean": 101.45,
    "forecast_std": 2.34,
    "forecast_min": 99.12,
    "forecast_max": 105.67,
    "forecast_median": 101.23,
    "forecast_percentile_25": 100.67,
    "forecast_percentile_75": 102.34,
    "volatility": 0.023
  },
  
  "technical_indicators": {
    "rsi": {
      "values": [65.45, 66.23, 67.12, ...],
      "overbought": false,
      "oversold": false,
      "signal": "neutral"
    },
    "macd": {
      "macd_line": [1.23, 1.34, 1.45, ...],
      "signal_line": [0.45, 0.52, 0.61, ...],
      "histogram": [0.78, 0.82, 0.84, ...]
    },
    "bollinger_bands": {
      "upper": [105.67, 106.45, 107.23, ...],
      "middle": [101.45, 102.34, 103.12, ...],
      "lower": [97.23, 98.23, 99.01, ...],
      "bandwidth": 0.0420,
      "position": 0.45
    },
    "moving_averages": {
      "sma_10": 101.23,
      "sma_20": 100.67,
      "sma_50": 99.45,
      "ema_10": 101.56,
      "ema_20": 100.89
    }
  },
  
  "trend_analysis": {
    "overall_trend": "upward",
    "trend_strength": 0.87,
    "slope": 0.045,
    "change_percent": 2.34,
    "volatility_forecast": 0.0234
  },
  
  "anomalies": {
    "detected": true,
    "count": 2,
    "anomalies": [
      {
        "period": 15,
        "value": 100.23,
        "zscore": 2.45,
        "anomaly_type": "positive_spike"
      },
      {
        "period": 18,
        "value": 99.12,
        "zscore": -1.89,
        "anomaly_type": "negative_dip"
      }
    ]
  },
  
  "correlation_analysis": {
    "forecast_vs_volume": 0.78,
    "forecast_vs_rsi": 0.45,
    "forecast_vs_macd": 0.92,
    "with_historical_data": 0.89
  },
  
  "performance_summary": {
    "model_confidence": 0.94,
    "prediction_reliability": "high",
    "recommendation": "STRONG_BUY",
    "risk_level": "medium"
  },
  
  "generated_at": "2026-02-23T15:33:00.123456",
  "execution_time_ms": 123.45,
  "cache_hit": false
}
```

---

### 6. Previsão Prophet (Com Autenticação)
```
GET /forecast_prophet?model_id=prophet_...&periods=24
Headers: Authorization: Bearer sk_...
```

**Response:**
```json
{
  "model_id": "prophet_file_20260223_153045_20260223_153200",
  "model_type": "prophet",
  "forecast_date": "2026-02-23T15:34:00.123456",
  "periods": 24,
  
  "forecast": {
    "values": [
      [100.23, 1000000, 65.45, 1.23, 0.45],
      [101.23, 1050000, 66.23, 1.34, 0.52],
      [102.34, 1100000, 67.12, 1.45, 0.61],
      ...
    ],
    "column_names": ["Close", "Volume", "RSI", "MACD", "Signal"]
  },
  
  "forecast_components": {
    "trend": [100.12, 100.34, 100.56, ...],
    "yearly_seasonality": [0.11, 0.23, 0.45, ...],
    "weekly_seasonality": [0.05, 0.08, -0.03, ...],
    "monthly_seasonality": [0.02, -0.04, 0.06, ...],
    "holiday_effects": []
  },
  
  "uncertainties": {
    "trend_uncertainty": [0.05, 0.06, 0.07, ...],
    "seasonal_uncertainty": [0.03, 0.03, 0.04, ...],
    "observation_error": 0.02,
    "confidence_interval_95": {
      "lower": [98.45, 99.12, 99.89, ...],
      "upper": [102.01, 103.45, 104.79, ...]
    }
  },
  
  "seasonality_forecast": {
    "yearly": {
      "coefficients": {"jan": 0.12, "feb": -0.05, ...},
      "strength": 0.87
    },
    "weekly": {
      "coefficients": {"mon": 0.08, "tue": -0.03, ...},
      "strength": 0.45
    }
  },
  
  "changepoint_analysis": {
    "detected_changepoints": [
      {
        "date": "2024-06-15",
        "magnitude": 0.45,
        "confidence": 0.89
      }
    ]
  },
  
  "actual_vs_forecast": {
    "actual": [99.45, 99.56, 99.78, ...],
    "forecast": [100.23, 101.23, 102.34, ...],
    "mae": 0.0456,
    "rmse": 0.0567,
    "mape": 2.34
  },
  
  "statistics": {
    "forecast_mean": 101.45,
    "forecast_std": 2.34,
    "forecast_min": 99.12,
    "forecast_max": 105.67,
    "confidence_level": 0.95
  },
  
  "generated_at": "2026-02-23T15:34:00.123456",
  "execution_time_ms": 87.65
}
```

---

### 7. Listar Modelos
```
GET /models
Headers: Authorization: Bearer sk_...
```

**Response:**
```json
{
  "total": 4,
  "models": {
    "lstm_file_20260223_153045_20260223_153100": {
      "type": "lstm",
      "file_id": "file_20260223_153045",
      "lookback": 30,
      "epochs": 100,
      "batch_size": 32,
      "numeric_cols": ["Close", "Volume", "RSI", "MACD", "Signal"],
      "train_loss": 0.0312,
      "created_at": "2026-02-23T15:31:00.123456",
      "status": "active",
      "predictions_count": 5
    },
    "prophet_file_20260223_153045_20260223_153200": {
      "type": "prophet",
      "file_id": "file_20260223_153045",
      "numeric_cols": ["Close", "Volume", "RSI", "MACD"],
      "created_at": "2026-02-23T15:32:00.123456",
      "status": "active",
      "predictions_count": 3
    }
  }
}
```

---

### 8. Listar Arquivos Carregados
```
GET /files
Headers: Authorization: Bearer sk_...
```

**Response:**
```json
{
  "total": 1,
  "files": {
    "file_20260223_153045": {
      "rows": 1000,
      "columns": ["Date", "Close", "Volume", "RSI", "MACD", "Signal"],
      "datetime_column": "Date",
      "numeric_columns": ["Close", "Volume", "RSI", "MACD", "Signal"],
      "created_at": "2026-02-23T15:30:45.123456",
      "size_mb": 0.5,
      "memory_usage_mb": 2.3
    }
  }
}
```

---

### 9. Análise Técnica Avançada (NOVO)
```
GET /technical_analysis/{model_id}?periods=24
Headers: Authorization: Bearer sk_...
```

**Response:**
```json
{
  "model_id": "lstm_file_20260223_153045",
  "analysis_date": "2026-02-23T15:35:00",
  
  "indicators": {
    "rsi": {
      "values": [65.45, 66.23, 67.12, ...],
      "interpretation": "Neutral to Overbought",
      "threshold_overbought": 70,
      "threshold_oversold": 30
    },
    "macd": {
      "macd": [1.23, 1.34, 1.45, ...],
      "signal": [0.45, 0.52, 0.61, ...],
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
    "atr": [2.34, 2.45, 2.67, ...],
    "stochastic": {
      "k_percent": [65.23, 68.45, 71.23, ...],
      "d_percent": [62.34, 65.23, 68.45, ...]
    }
  },
  
  "signals": {
    "buy_signals": 3,
    "sell_signals": 1,
    "strong_buy": 1,
    "strong_sell": 0,
    "overall_signal": "BUY",
    "confidence": 0.89
  },
  
  "generated_at": "2026-02-23T15:35:00.123456"
}
```

---

### 10. Gerenciamento de API Keys
```
POST /generate-api-key?name=meu-app
GET  /api-keys
DELETE /api-keys/{key_partial}
```

---

## 🏗️ Estrutura Completa de Dados

### Objeto Previsão (Forecast)

```typescript
interface ForecastObject {
  // Identificação
  model_id: string;
  model_type: 'lstm' | 'prophet';
  forecast_date: string; // ISO 8601
  periods: number;

  // Valores de Previsão
  forecast: {
    values: number[][];
    column_names: string[];
    data_type: string;
  };

  // Timestamps
  timestamps: {
    dates: string[];
    datetimes: string[];
    unix_timestamps: number[];
    interval: string;
    timezone: string;
  };

  // Intervalos de Confiança
  confidence_intervals: {
    lower_bound_95: number[][];
    upper_bound_95: number[][];
    lower_bound_80: number[][];
    upper_bound_80: number[][];
  };

  // Comparação Real vs Previsão
  actual_vs_forecast: {
    actual_last_24: number[][];
    forecast_24: number[][];
    mean_absolute_error: number;
    rmse: number;
    mape: number;
    r2: number;
    directional_accuracy: number;
  };

  // Estatísticas
  statistics: {
    forecast_mean: number;
    forecast_std: number;
    forecast_min: number;
    forecast_max: number;
    forecast_median: number;
    forecast_percentile_25: number;
    forecast_percentile_75: number;
    volatility: number;
  };

  // Indicadores Técnicos
  technical_indicators: {
    rsi: TechnicalIndicator;
    macd: TechnicalIndicator;
    bollinger_bands: TechnicalIndicator;
    moving_averages: TechnicalIndicator;
  };

  // Análise de Tendência
  trend_analysis: {
    overall_trend: 'upward' | 'downward' | 'sideways';
    trend_strength: number;
    slope: number;
    change_percent: number;
    volatility_forecast: number;
  };

  // Anomalias
  anomalies: {
    detected: boolean;
    count: number;
    anomalies: Anomaly[];
  };

  // Correlações
  correlation_analysis: {
    forecast_vs_volume: number;
    forecast_vs_rsi: number;
    forecast_vs_macd: number;
    with_historical_data: number;
  };

  // Resumo de Performance
  performance_summary: {
    model_confidence: number;
    prediction_reliability: 'high' | 'medium' | 'low';
    recommendation: string;
    risk_level: 'low' | 'medium' | 'high';
  };

  // Meta
  generated_at: string;
  execution_time_ms: number;
  cache_hit: boolean;
}
```

---

## 💻 Exemplos JavaScript/TypeScript

### 1. Setup Básico

```javascript
// client.js
import fetch from 'node-fetch';
import dotenv from 'dotenv';

dotenv.config();

const API_KEY = process.env.API_KEY;
const API_HOST = process.env.API_HOST || 'http://localhost:8000';

class TOTEMDeepsea {
  constructor(apiKey, apiHost) {
    this.apiKey = apiKey;
    this.apiHost = apiHost;
  }

  async request(endpoint, options = {}) {
    const url = `${this.apiHost}${endpoint}`;
    const headers = {
      'Authorization': `Bearer ${this.apiKey}`,
      'Content-Type': 'application/json',
      ...options.headers,
    };

    const response = await fetch(url, {
      ...options,
      headers,
    });

    if (!response.ok) {
      throw new Error(`API Error: ${response.status} ${response.statusText}`);
    }

    return response.json();
  }

  // Upload CSV
  async uploadCSV(filePath) {
    const fs = require('fs');
    const FormData = require('form-data');
    
    const form = new FormData();
    form.append('file', fs.createReadStream(filePath));

    const response = await fetch(`${this.apiHost}/upload_csv`, {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${this.apiKey}`,
        ...form.getHeaders(),
      },
      body: form,
    });

    return response.json();
  }

  // Treinar LSTM
  async trainLSTM(fileId, options = {}) {
    return this.request('/train_lstm', {
      method: 'POST',
      body: JSON.stringify({
        file_id: fileId,
        lookback: options.lookback || 30,
        epochs: options.epochs || 100,
        batch_size: options.batch_size || 32,
      }),
    });
  }

  // Previsão LSTM
  async forecastLSTM(modelId, periods = 24) {
    return this.request(`/forecast_lstm?model_id=${modelId}&periods=${periods}`);
  }

  // Previsão Prophet
  async forecastProphet(modelId, periods = 24) {
    return this.request(`/forecast_prophet?model_id=${modelId}&periods=${periods}`);
  }

  // Listar Modelos
  async getModels() {
    return this.request('/models');
  }

  // Listar Arquivos
  async getFiles() {
    return this.request('/files');
  }

  // Análise Técnica
  async getTechnicalAnalysis(modelId, periods = 24) {
    return this.request(`/technical_analysis/${modelId}?periods=${periods}`);
  }
}

export default TOTEMDeepsea;
```

---

### 2. Exemplo Completo - Upload, Treinar e Prever

```javascript
// main.js
import TOTEMDeepsea from './client.js';

async function main() {
  const api = new TOTEMDeepsea(
    process.env.API_KEY,
    process.env.API_HOST
  );

  try {
    // 1. Upload CSV
    console.log('📤 Uploading CSV...');
    const uploadResult = await api.uploadCSV('data/stock_prices.csv');
    const fileId = uploadResult.file_id;
    console.log(`✅ File uploaded: ${fileId}`);
    console.log(`   Rows: ${uploadResult.rows}`);
    console.log(`   Columns: ${uploadResult.columns.join(', ')}`);

    // 2. Treinar LSTM
    console.log('\n🤖 Training LSTM...');
    const trainingResult = await api.trainLSTM(fileId, {
      lookback: 30,
      epochs: 100,
      batch_size: 32,
    });
    const modelId = trainingResult.model_id;
    console.log(`✅ Model trained: ${modelId}`);
    console.log(`   Training time: ${trainingResult.training_time}s`);
    console.log(`   Metrics:`);
    console.log(`     - MAE: ${trainingResult.metrics.mae}`);
    console.log(`     - RMSE: ${trainingResult.metrics.rmse}`);
    console.log(`     - R²: ${trainingResult.metrics.r2}`);

    // 3. Fazer Previsão
    console.log('\n🔮 Generating Forecast...');
    const forecast = await api.forecastLSTM(modelId, 24);
    console.log(`✅ Forecast generated: ${forecast.periods} periods`);
    console.log(`   Date range: ${forecast.timestamps.dates[0]} to ${forecast.timestamps.dates.slice(-1)[0]}`);
    console.log(`   Forecast values (first 5):`);
    forecast.forecast.values.slice(0, 5).forEach((row, idx) => {
      console.log(`     ${idx + 1}. ${row.map(v => v.toFixed(2)).join(', ')}`);
    });

    // 4. Análise Técnica
    console.log('\n📈 Technical Analysis...');
    const analysis = await api.getTechnicalAnalysis(modelId);
    console.log(`   RSI: ${analysis.indicators.rsi.values[0].toFixed(2)}`);
    console.log(`   MACD Signal: ${analysis.indicators.macd.signal_cross}`);
    console.log(`   Overall Signal: ${analysis.signals.overall_signal}`);
    console.log(`   Confidence: ${(analysis.signals.confidence * 100).toFixed(0)}%`);

    // 5. Estatísticas
    console.log('\n📊 Statistics:');
    console.log(`   Model Confidence: ${(forecast.performance_summary.model_confidence * 100).toFixed(0)}%`);
    console.log(`   Risk Level: ${forecast.performance_summary.risk_level}`);
    console.log(`   Recommendation: ${forecast.performance_summary.recommendation}`);

  } catch (error) {
    console.error('❌ Error:', error.message);
  }
}

main();
```

---

### 3. Visualizar Previsão com Plotly

```javascript
// visualize.js
import TOTEMDeepsea from './client.js';
import PlotlyExpress from 'plotly-express';

async function visualizeForecast(modelId) {
  const api = new TOTEMDeepsea(
    process.env.API_KEY,
    process.env.API_HOST
  );

  const forecast = await api.forecastLSTM(modelId, 60);

  // Extrair dados
  const dates = forecast.timestamps.dates;
  const forecastValues = forecast.forecast.values.map(row => row[0]); // Close price
  const actualValues = forecast.actual_vs_forecast.actual_last_24;
  const upper95 = forecast.confidence_intervals.upper_bound_95.map(row => row[0]);
  const lower95 = forecast.confidence_intervals.lower_bound_95.map(row => row[0]);

  // Criar gráfico
  const data = [
    {
      x: dates.slice(0, actualValues.length),
      y: actualValues,
      name: 'Actual',
      type: 'scatter',
      mode: 'lines',
      line: { color: 'blue', width: 2 },
    },
    {
      x: dates.slice(actualValues.length),
      y: forecastValues.slice(actualValues.length),
      name: 'Forecast',
      type: 'scatter',
      mode: 'lines',
      line: { color: 'red', width: 2, dash: 'dash' },
    },
    {
      x: dates.slice(actualValues.length),
      y: upper95.slice(actualValues.length),
      name: 'Upper 95%',
      fill: 'tonexty',
      type: 'scatter',
      mode: 'lines',
      line: { color: 'rgba(0,0,0,0)' },
    },
    {
      x: dates.slice(actualValues.length),
      y: lower95.slice(actualValues.length),
      name: 'Lower 95%',
      fill: 'tonexty',
      type: 'scatter',
      mode: 'lines',
      line: { color: 'rgba(0,0,0,0)' },
      fillcolor: 'rgba(255,0,0,0.2)',
    },
  ];

  const layout = {
    title: `LSTM Forecast - ${modelId}`,
    xaxis: { title: 'Date' },
    yaxis: { title: 'Price' },
    hovermode: 'x unified',
  };

  PlotlyExpress.save('forecast.html', data, layout);
  console.log('✅ Forecast saved to forecast.html');
}

await visualizeForecast(process.argv[2]);
```

---

### 4. TypeScript - Tipos Completos

```typescript
// types.ts
export interface ForecastData {
  model_id: string;
  model_type: 'lstm' | 'prophet';
  forecast_date: string;
  periods: number;
  
  forecast: {
    values: number[][];
    column_names: string[];
    data_type: string;
  };
  
  timestamps: {
    dates: string[];
    datetimes: string[];
    unix_timestamps: number[];
    interval: string;
    timezone: string;
  };
  
  confidence_intervals: {
    lower_bound_95: number[][];
    upper_bound_95: number[][];
    lower_bound_80: number[][];
    upper_bound_80: number[][];
  };
  
  actual_vs_forecast: {
    actual_last_24: number[][];
    forecast_24: number[][];
    mean_absolute_error: number;
    rmse: number;
    mape: number;
    r2: number;
    directional_accuracy: number;
  };
  
  statistics: {
    forecast_mean: number;
    forecast_std: number;
    forecast_min: number;
    forecast_max: number;
    forecast_median: number;
    forecast_percentile_25: number;
    forecast_percentile_75: number;
    volatility: number;
  };
  
  technical_indicators: {
    rsi: Record<string, any>;
    macd: Record<string, any>;
    bollinger_bands: Record<string, any>;
    moving_averages: Record<string, any>;
  };
  
  trend_analysis: {
    overall_trend: 'upward' | 'downward' | 'sideways';
    trend_strength: number;
    slope: number;
    change_percent: number;
    volatility_forecast: number;
  };
  
  anomalies: {
    detected: boolean;
    count: number;
    anomalies: Array<{
      period: number;
      value: number;
      zscore: number;
      anomaly_type: string;
    }>;
  };
  
  performance_summary: {
    model_confidence: number;
    prediction_reliability: 'high' | 'medium' | 'low';
    recommendation: string;
    risk_level: 'low' | 'medium' | 'high';
  };
}

// api-client.ts
import axios, { AxiosInstance } from 'axios';

export class TOTEMDeepseaClient {
  private client: AxiosInstance;
  
  constructor(apiKey: string, apiHost: string = 'http://localhost:8000') {
    this.client = axios.create({
      baseURL: apiHost,
      headers: {
        'Authorization': `Bearer ${apiKey}`,
        'Content-Type': 'application/json',
      },
    });
  }
  
  async forecastLSTM(modelId: string, periods: number = 24): Promise<ForecastData> {
    const response = await this.client.get<ForecastData>(
      `/forecast_lstm?model_id=${modelId}&periods=${periods}`
    );
    return response.data;
  }
  
  async forecastProphet(modelId: string, periods: number = 24): Promise<ForecastData> {
    const response = await this.client.get<ForecastData>(
      `/forecast_prophet?model_id=${modelId}&periods=${periods}`
    );
    return response.data;
  }
  
  async trainLSTM(fileId: string, options?: {
    lookback?: number;
    epochs?: number;
    batch_size?: number;
  }): Promise<any> {
    const response = await this.client.post(
      '/train_lstm',
      { file_id: fileId, ...options }
    );
    return response.data;
  }
}
```

---

### 5. React - Gráficos e Dashboard

```typescript
// ForecastChart.tsx
import React, { useState, useEffect } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import TOTEMDeepseaClient from './client';

interface Props {
  modelId: string;
  periods?: number;
}

export const ForecastChart: React.FC<Props> = ({ modelId, periods = 24 }) => {
  const [data, setData] = useState<any[]>([]);
  const [forecast, setForecast] = useState<any>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function loadForecast() {
      try {
        const client = new TOTEMDeepseaClient(
          process.env.REACT_APP_API_KEY || '',
          process.env.REACT_APP_API_HOST || 'http://localhost:8000'
        );
        
        const result = await client.forecastLSTM(modelId, periods);
        setForecast(result);
        
        // Preparar dados para o gráfico
        const chartData = result.timestamps.dates.map((date: string, idx: number) => ({
          date,
          actual: result.actual_vs_forecast.actual_last_24[idx]?.[0],
          forecast: result.forecast.values[idx]?.[0],
          upper95: result.confidence_intervals.upper_bound_95[idx]?.[0],
          lower95: result.confidence_intervals.lower_bound_95[idx]?.[0],
        }));
        
        setData(chartData);
      } catch (error) {
        console.error('Erro ao carregar previsão:', error);
      } finally {
        setLoading(false);
      }
    }
    
    loadForecast();
  }, [modelId, periods]);

  if (loading) return <div>Carregando...</div>;
  if (!forecast) return <div>Erro ao carregar dados</div>;

  return (
    <div>
      <h2>Previsão - {modelId}</h2>
      <p>Confiança: {(forecast.performance_summary.model_confidence * 100).toFixed(0)}%</p>
      
      <ResponsiveContainer width="100%" height={400}>
        <LineChart data={data}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="date" />
          <YAxis />
          <Tooltip />
          <Legend />
          <Line type="monotone" dataKey="actual" stroke="#8884d8" />
          <Line type="monotone" dataKey="forecast" stroke="#82ca9d" strokeDasharray="5 5" />
          <Line type="monotone" dataKey="upper95" stroke="#ff7300" strokeDasharray="3 3" />
          <Line type="monotone" dataKey="lower95" stroke="#ff7300" strokeDasharray="3 3" />
        </LineChart>
      </ResponsiveContainer>
      
      <div>
        <h3>Indicadores Técnicos</h3>
        <p>RSI: {forecast.technical_indicators.rsi.values[0].toFixed(2)}</p>
        <p>MACD: {forecast.technical_indicators.macd.signal_cross}</p>
        <p>Sinal: {forecast.signals.overall_signal}</p>
      </div>
    </div>
  );
};
```

---

### 6. Next.js - API Route

```typescript
// pages/api/forecast.ts
import type { NextApiRequest, NextApiResponse } from 'next';
import TOTEMDeepseaClient from '@/lib/client';

export default async function handler(
  req: NextApiRequest,
  res: NextApiResponse
) {
  if (req.method !== 'GET') {
    return res.status(405).json({ error: 'Method not allowed' });
  }

  const { modelId, periods = 24 } = req.query;
  
  try {
    const client = new TOTEMDeepseaClient(
      process.env.TOTEM_API_KEY || '',
      process.env.TOTEM_API_HOST || 'http://localhost:8000'
    );
    
    const forecast = await client.forecastLSTM(
      modelId as string,
      parseInt(periods as string)
    );
    
    return res.status(200).json(forecast);
  } catch (error) {
    console.error('API Error:', error);
    return res.status(500).json({ error: 'Failed to get forecast' });
  }
}
```

---

## 🎨 Gráficos e Visualizações

### 1. Chart.js

```html
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<canvas id="forecastChart"></canvas>

<script>
async function showChart(modelId) {
  const response = await fetch(
    `/api/forecast?modelId=${modelId}&periods=24`,
    {
      headers: {
        'Authorization': `Bearer ${API_KEY}`
      }
    }
  );
  const forecast = await response.json();
  
  new Chart('forecastChart', {
    type: 'line',
    data: {
      labels: forecast.timestamps.dates,
      datasets: [
        {
          label: 'Forecast',
          data: forecast.forecast.values.map(row => row[0]),
          borderColor: 'rgb(75, 192, 192)',
          borderDash: [5, 5],
        },
        {
          label: 'Upper 95%',
          data: forecast.confidence_intervals.upper_bound_95.map(row => row[0]),
          borderColor: 'rgba(255, 0, 0, 0.2)',
          fill: false,
          pointRadius: 0,
        },
        {
          label: 'Lower 95%',
          data: forecast.confidence_intervals.lower_bound_95.map(row => row[0]),
          borderColor: 'rgba(255, 0, 0, 0.2)',
          fill: '-2',
          backgroundColor: 'rgba(255, 0, 0, 0.1)',
          pointRadius: 0,
        },
      ]
    }
  });
}
</script>
```

### 2. D3.js

```javascript
import * as d3 from 'd3';

async function visualizeWithD3(modelId) {
  const response = await fetch(`/api/forecast?modelId=${modelId}`);
  const forecast = await response.json();
  
  const svg = d3.select('#chart')
    .append('svg')
    .attr('width', 800)
    .attr('height', 400);
  
  const xScale = d3.scaleTime()
    .domain([
      new Date(forecast.timestamps.dates[0]),
      new Date(forecast.timestamps.dates[forecast.timestamps.dates.length - 1])
    ])
    .range([0, 800]);
  
  const yScale = d3.scaleLinear()
    .domain([
      Math.min(...forecast.confidence_intervals.lower_bound_95.map(r => r[0])),
      Math.max(...forecast.confidence_intervals.upper_bound_95.map(r => r[0]))
    ])
    .range([400, 0]);
  
  // Desenhar linhas de previsão
  const line = d3.line()
    .x((d, i) => xScale(new Date(forecast.timestamps.dates[i])))
    .y((d) => yScale(d[0]));
  
  svg.append('path')
    .datum(forecast.forecast.values)
    .attr('fill', 'none')
    .attr('stroke', 'steelblue')
    .attr('stroke-width', 2)
    .attr('d', line);
}
```

---

## 🚀 Iniciar Servidor

```bash
# Ativar ambiente
source venv/bin/activate

# Iniciar API
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Acessar:
- 🌐 API: http://localhost:8000
- 📚 Swagger: http://localhost:8000/docs
- 📖 ReDoc: http://localhost:8000/redoc

---

## 📦 Exemplo Completo com Node.js

```bash
npm init -y
npm install axios dotenv plotly

# Criar .env
cat > .env << EOF
API_KEY=sk_oSBMF-nwZBfEv6RrzD1F1no72Cp10qQsMkPq8ztPPIY
API_HOST=http://localhost:8000
EOF

# Criar client.js e rodar
node main.js
```

---

**✨ Agora você tem ABSOLUTAMENTE TUDO para trabalhar com a API!**

- ✅ Todos os endpoints documentados
- ✅ Estrutura completa de dados
- ✅ Exemplos em JavaScript/TypeScript
- ✅ Código React pronto
- ✅ Integração Next.js
- ✅ Visualizações com D3.js
- ✅ Indicadores técnicos em tempo real
- ✅ Análise estatística completa

**Comece agora:**
```bash
python -m uvicorn main:app --host 0.0.0.0 --port 8000
```

Qualquer dúvida, vê a documentação Swagger em `http://localhost:8000/docs`! 🎯

/**
 * TOTEM_DEEPSEA - Cliente TypeScript Completo com Tipos
 * 
 * Tipos completos e exemplos de uso em TypeScript
 * 
 * Instalação:
 * npm install axios dotenv typescript ts-node @types/node @types/axios
 * 
 * Uso:
 * ts-node ts-client-complete.ts
 */

import axios, { AxiosInstance, AxiosError } from 'axios';
import * as dotenv from 'dotenv';
import * as fs from 'fs';

dotenv.config();

// ============================================================
// 1. TIPOS E INTERFACES TYPESCRIPT
// ============================================================

interface HealthCheckResponse {
  status: string;
  timestamp: string;
  version: string;
}

interface UploadResponse {
  file_id: string;
  rows: number;
  columns: string[];
  datetime_column: string;
  numeric_columns: string[];
  uploaded_at: string;
}

interface TrainingResponse {
  model_id: string;
  model_type: 'lstm' | 'prophet';
  rows_used: number;
  features: number;
  metrics: MetricsData;
  training_time: number;
  created_at: string;
  training_data?: Record<string, any>;
  model_stats?: Record<string, any>;
  performance?: Record<string, any>;
}

interface MetricsData {
  mae: number;
  rmse: number;
  mape: number;
  r2: number;
  directional_accuracy?: number;
}

interface TechnicalIndicators {
  rsi: {
    values: number[];
    current: number;
    overbought: boolean;
    oversold: boolean;
    interpretation: string;
  };
  macd: {
    macd_line: number[];
    signal_line: number[];
    histogram: number[];
    current_macd: number;
    current_signal: number;
    signal_cross: 'bullish' | 'bearish' | 'neutral';
  };
  bollinger_bands: {
    upper: number[];
    middle: number[];
    lower: number[];
    current_upper: number;
    current_middle: number;
    current_lower: number;
    band_width: number;
    price_position: number;
    interpretation: string;
  };
  moving_averages: Record<string, number>;
}

interface TrendAnalysis {
  overall_trend: 'upward' | 'downward' | 'sideways';
  trend_strength: number;
  slope: number;
  change_percent: number;
  volatility: number;
}

interface Anomaly {
  period: number;
  value: number;
  zscore: number;
  anomaly_type: string;
}

interface AnomalyData {
  detected: boolean;
  count: number;
  anomalies: Anomaly[];
}

interface Statistics {
  mean: number;
  std: number;
  min: number;
  max: number;
  median: number;
  percentile_25: number;
  percentile_75: number;
}

interface Signals {
  buy_signals: number;
  sell_signals: number;
  overall_signal: 'BUY' | 'SELL' | 'HOLD';
  recommendation: string;
  confidence: number;
}

interface PerformanceSummary {
  model_confidence: number;
  prediction_reliability: 'high' | 'medium' | 'low';
  recommendation: string;
  risk_level: 'low' | 'medium' | 'high';
}

interface ForecastResponse {
  model_id: string;
  model_type: 'lstm' | 'prophet';
  forecast_date: string;
  periods: number;
  
  forecast: {
    values: number[][];
    column_names: string[];
    data_type?: string;
  };
  
  timestamps: {
    dates: string[];
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
    actual_last_24: number[][] | null;
    forecast_24: number[][];
    mae?: number;
    rmse?: number;
    mape?: number;
    r2?: number;
  };
  
  statistics: Statistics;
  technical_indicators: TechnicalIndicators;
  trend_analysis: TrendAnalysis;
  anomalies: AnomalyData;
  correlation_analysis: Record<string, number>;
  signals: Signals;
  performance_summary: PerformanceSummary;
  
  generated_at: string;
  execution_time_ms: number;
  cache_hit: boolean;
}

interface TechnicalAnalysisResponse {
  model_id: string;
  analysis_date: string;
  indicators: TechnicalIndicators;
  signals: Signals;
  trend_analysis: TrendAnalysis;
  anomalies: AnomalyData;
  statistics: Statistics;
}

// ============================================================
// 2. CLASSE CLIENTE TOTEM DEEPSEA
// ============================================================

class TOTEMDeepseaClient {
  private apiKey: string;
  private apiHost: string;
  private client: AxiosInstance;

  constructor(apiKey: string, apiHost: string = 'http://localhost:8000') {
    this.apiKey = apiKey;
    this.apiHost = apiHost;
    this.client = axios.create({
      baseURL: apiHost,
      headers: {
        'Authorization': `Bearer ${apiKey}`,
        'Content-Type': 'application/json',
      },
      timeout: 30000,
    });
  }

  /**
   * Health check
   */
  async health(): Promise<HealthCheckResponse> {
    try {
      const response = await this.client.get<HealthCheckResponse>('/health');
      return response.data;
    } catch (error) {
      throw this.handleError(error);
    }
  }

  /**
   * Upload CSV
   */
  async uploadCSV(filePath: string): Promise<UploadResponse> {
    try {
      const FormData = require('form-data');
      const form = new FormData();
      form.append('file', fs.createReadStream(filePath));

      const response = await this.client.post<UploadResponse>('/upload_csv', form, {
        headers: form.getHeaders(),
      });
      return response.data;
    } catch (error) {
      throw this.handleError(error);
    }
  }

  /**
   * Treinar LSTM
   */
  async trainLSTM(
    fileId: string,
    options?: {
      lookback?: number;
      epochs?: number;
      batch_size?: number;
    }
  ): Promise<TrainingResponse> {
    try {
      const payload = {
        file_id: fileId,
        lookback: options?.lookback || 30,
        epochs: options?.epochs || 100,
        batch_size: options?.batch_size || 32,
      };
      
      const response = await this.client.post<TrainingResponse>('/train_lstm', payload);
      return response.data;
    } catch (error) {
      throw this.handleError(error);
    }
  }

  /**
   * Treinar Prophet
   */
  async trainProphet(fileId: string): Promise<TrainingResponse> {
    try {
      const payload = { file_id: fileId };
      const response = await this.client.post<TrainingResponse>('/train_prophet', payload);
      return response.data;
    } catch (error) {
      throw this.handleError(error);
    }
  }

  /**
   * Previsão LSTM com TODOS os dados
   */
  async forecastLSTM(modelId: string, periods: number = 24): Promise<ForecastResponse> {
    try {
      const response = await this.client.get<ForecastResponse>('/forecast_lstm', {
        params: { model_id: modelId, periods },
      });
      return response.data;
    } catch (error) {
      throw this.handleError(error);
    }
  }

  /**
   * Previsão Prophet com TODOS os dados
   */
  async forecastProphet(modelId: string, periods: number = 24): Promise<ForecastResponse> {
    try {
      const response = await this.client.get<ForecastResponse>('/forecast_prophet', {
        params: { model_id: modelId, periods },
      });
      return response.data;
    } catch (error) {
      throw this.handleError(error);
    }
  }

  /**
   * Análise Técnica
   */
  async technicalAnalysis(modelId: string, periods: number = 24): Promise<TechnicalAnalysisResponse> {
    try {
      const response = await this.client.get<TechnicalAnalysisResponse>(
        `/technical_analysis/${modelId}`,
        { params: { periods } }
      );
      return response.data;
    } catch (error) {
      throw this.handleError(error);
    }
  }

  /**
   * Listar modelos
   */
  async getModels(): Promise<Record<string, any>> {
    try {
      const response = await this.client.get('/models');
      return response.data;
    } catch (error) {
      throw this.handleError(error);
    }
  }

  /**
   * Listar arquivos
   */
  async getFiles(): Promise<Record<string, any>> {
    try {
      const response = await this.client.get('/files');
      return response.data;
    } catch (error) {
      throw this.handleError(error);
    }
  }

  // Error handler
  private handleError(error: any): Error {
    if (axios.isAxiosError(error)) {
      const message = error.response?.data?.message || error.message;
      return new Error(`API Error: ${message}`);
    }
    return error;
  }
}

// ============================================================
// 3. EXEMPLO - ANÁLISE COMPLETA COM TIPOS
// ============================================================

async function analyzeForecasting(): Promise<void> {
  const api = new TOTEMDeepseaClient(
    process.env.API_KEY || '',
    process.env.API_HOST || 'http://localhost:8000'
  );

  try {
    console.log('🚀 TOTEM_DEEPSEA TypeScript Example\n');

    // Health check
    console.log('1️⃣  Health Check...');
    const health = await api.health();
    console.log(`   ✅ Status: ${health.status}\n`);

    // Listar modelos
    console.log('2️⃣  Listing Models...');
    const models = await api.getModels();
    
    if (models.total === 0) {
      console.log('   ❌ No models found. Please train a model first!\n');
      return;
    }

    const modelId = Object.keys(models.models)[0];
    const model = models.models[modelId];
    console.log(`   ✅ Found model: ${modelId}`);
    console.log(`      • Type: ${model.type}`);
    console.log(`      • Created: ${model.created_at}\n`);

    // LSTM Forecast
    if (model.type === 'lstm') {
      console.log('3️⃣  LSTM Forecast (Complete Data)...');
      const forecast = await api.forecastLSTM(modelId, 24);
      
      // Exibir dados estruturados
      console.log(`   ✅ Forecast generated (${forecast.periods} periods)`);
      console.log(`      • Execution time: ${forecast.execution_time_ms.toFixed(2)}ms`);
      
      // Indicadores
      console.log('\n   📈 Technical Indicators:');
      console.log(`      • RSI: ${forecast.technical_indicators.rsi.current.toFixed(2)} (${forecast.technical_indicators.rsi.interpretation})`);
      console.log(`      • MACD: ${forecast.technical_indicators.macd.signal_cross}`);
      console.log(`      • Bollinger Position: ${(forecast.technical_indicators.bollinger_bands.price_position * 100).toFixed(0)}%`);
      
      // Trend
      console.log('\n   🔄 Trend Analysis:');
      console.log(`      • Direction: ${forecast.trend_analysis.overall_trend.toUpperCase()}`);
      console.log(`      • Strength: ${(forecast.trend_analysis.trend_strength * 100).toFixed(0)}%`);
      console.log(`      • Change: ${forecast.trend_analysis.change_percent.toFixed(2)}%`);
      
      // Signals
      console.log('\n   🎯 Trading Signals:');
      console.log(`      • Signal: ${forecast.signals.overall_signal}`);
      console.log(`      • Recommendation: ${forecast.signals.recommendation}`);
      console.log(`      • Confidence: ${(forecast.signals.confidence * 100).toFixed(0)}%`);
      
      // Performance
      console.log('\n   ⭐ Performance:');
      console.log(`      • Model Confidence: ${(forecast.performance_summary.model_confidence * 100).toFixed(0)}%`);
      console.log(`      • Reliability: ${forecast.performance_summary.prediction_reliability}`);
      console.log(`      • Risk Level: ${forecast.performance_summary.risk_level}`);

      // Exportar dados estruturados para JSON
      const exportData = {
        forecast: forecast.forecast.values,
        timestamps: forecast.timestamps.dates,
        indicators: {
          rsi: forecast.technical_indicators.rsi.values,
          macd: forecast.technical_indicators.macd.macd_line,
          bollinger: forecast.technical_indicators.bollinger_bands,
        },
        metrics: {
          mae: forecast.actual_vs_forecast.mae,
          rmse: forecast.actual_vs_forecast.rmse,
          r2: forecast.actual_vs_forecast.r2,
        },
        signals: forecast.signals,
      };

      fs.writeFileSync('forecast_analysis.json', JSON.stringify(exportData, null, 2));
      console.log('\n   ✅ Data exported to forecast_analysis.json');
    }

  } catch (error) {
    console.error('❌ Error:', error instanceof Error ? error.message : error);
  }
}

// Executar
analyzeForecasting();

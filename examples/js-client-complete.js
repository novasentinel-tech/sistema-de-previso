/**
 * TOTEM_DEEPSEA - Cliente JavaScript/TypeScript Completo
 * 
 * Exemplos de uso da API com TODOS os dados e indicadores técnicos
 * 
 * Instalação:
 * npm install axios dotenv plotly.js recharts
 * 
 * Uso:
 * node js-client-complete.js
 */

const dotenv = require('dotenv');
const axios = require('axios');

dotenv.config();

// ============================================================
// 1. CLASSE CLIENTE TOTEM DEEPSEA
// ============================================================

class TOTEMDeepseaClient {
  constructor(apiKey, apiHost = 'http://localhost:8000') {
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
  async health() {
    const response = await this.client.get('/health');
    return response.data;
  }

  /**
   * Upload CSV
   */
  async uploadCSV(filePath) {
    const fs = require('fs');
    const FormData = require('form-data');

    const form = new FormData();
    form.append('file', fs.createReadStream(filePath));

    const response = await this.client.post('/upload_csv', form, {
      headers: form.getHeaders(),
    });
    return response.data;
  }

  /**
   * Treinar LSTM
   */
  async trainLSTM(fileId, options = {}) {
    const payload = {
      file_id: fileId,
      lookback: options.lookback || 30,
      epochs: options.epochs || 100,
      batch_size: options.batch_size || 32,
    };
    
    const response = await this.client.post('/train_lstm', payload);
    return response.data;
  }

  /**
   * Treinar Prophet
   */
  async trainProphet(fileId, options = {}) {
    const payload = {
      file_id: fileId,
      quarterly_seasonality: options.quarterly_seasonality !== false,
      yearly_seasonality: options.yearly_seasonality !== false,
      interval_width: options.interval_width || 0.95,
    };
    
    const response = await this.client.post('/train_prophet', payload);
    return response.data;
  }

  /**
   * Previsão LSTM com TODOS os dados
   */
  async forecastLSTM(modelId, periods = 24) {
    const response = await this.client.get('/forecast_lstm', {
      params: { model_id: modelId, periods },
    });
    return response.data;
  }

  /**
   * Previsão Prophet com TODOS os dados
   */
  async forecastProphet(modelId, periods = 24) {
    const response = await this.client.get('/forecast_prophet', {
      params: { model_id: modelId, periods },
    });
    return response.data;
  }

  /**
   * Análise Técnica
   */
  async technicalAnalysis(modelId, periods = 24) {
    const response = await this.client.get(`/technical_analysis/${modelId}`, {
      params: { periods },
    });
    return response.data;
  }

  /**
   * Listar modelos
   */
  async getModels() {
    const response = await this.client.get('/models');
    return response.data;
  }

  /**
   * Listar arquivos
   */
  async getFiles() {
    const response = await this.client.get('/files');
    return response.data;
  }
}

// ============================================================
// 2. EXEMPLO COMPLETO - UPLOAD, TRAIN, FORECAST
// ============================================================

async function exampleCompleteWorkflow() {
  console.log('\n' + '='.repeat(60));
  console.log('🚀 TOTEM_DEEPSEA - Workflow Completo');
  console.log('='.repeat(60));

  const api = new TOTEMDeepseaClient(
    process.env.API_KEY,
    process.env.API_HOST || 'http://localhost:8000'
  );

  try {
    // 1. Health Check
    console.log('\n1️⃣  Health Check...');
    const health = await api.health();
    console.log('   ✅ API Status:', health.status);

    // 2. Upload CSV
    console.log('\n2️⃣  Uploading CSV...');
    // Criar arquivo de exemplo
    const fs = require('fs');
    const sampleData = `Date,Close,Volume,RSI,MACD
2024-01-01,100.00,1000000,50.0,0.5
2024-01-02,101.50,1100000,52.3,0.52
2024-01-03,99.80,950000,48.5,0.48
2024-01-04,102.30,1200000,55.2,0.55`;
    
    fs.writeFileSync('sample.csv', sampleData);
    
    const uploadResult = await api.uploadCSV('sample.csv');
    const fileId = uploadResult.file_id;
    console.log('   ✅ File uploaded:', fileId);
    console.log('      • Rows:', uploadResult.rows);
    console.log('      • Columns:', uploadResult.columns.join(', '));

    // 3. Treinar LSTM
    console.log('\n3️⃣  Training LSTM...');
    const lstmResult = await api.trainLSTM(fileId, {
      lookback: 10,
      epochs: 50,
      batch_size: 4,
    });
    const lstmModelId = lstmResult.model_id;
    
    console.log('   ✅ LSTM Model trained:', lstmModelId);
    console.log('      • Training time:', lstmResult.training_time, 'seconds');
    console.log('      • Epochs:', lstmResult.training_data.epochs);
    console.log('      • Parameters:', lstmResult.model_stats.total_parameters);
    console.log('      • Metrics:');
    console.log('        - MAE:', lstmResult.metrics.mae.toFixed(6));
    console.log('        - RMSE:', lstmResult.metrics.rmse.toFixed(6));
    console.log('        - R²:', lstmResult.metrics.r2.toFixed(4));

    // 4. Treinar Prophet
    console.log('\n4️⃣  Training Prophet...');
    const prophetResult = await api.trainProphet(fileId);
    const prophetModelId = prophetResult.model_id;
    
    console.log('   ✅ Prophet Model trained:', prophetModelId);
    console.log('      • Training time:', prophetResult.training_time, 'seconds');
    console.log('      • Seasonality detected:', 
        prophetResult.seasonality_analysis ? 'Yes' : 'No');

    // 5. Fazer Previsão LSTM com TODOS os dados
    console.log('\n5️⃣  LSTM Forecast (with COMPLETE data)...');
    const lstmForecast = await api.forecastLSTM(lstmModelId, 10);
    
    console.log('   ✅ LSTM Forecast generated');
    console.log('      • Periods:', lstmForecast.periods);
    console.log('      • Execution time:', lstmForecast.execution_time_ms.toFixed(2), 'ms');
    
    // Dados de Previsão
    console.log('\n   📊 Forecast Values (primeiras 3):');
    lstmForecast.forecast.values.slice(0, 3).forEach((row, idx) => {
      console.log(`      [${idx}] ${row.map(v => v.toFixed(2)).join(', ')}`);
    });
    
    // Indicadores Técnicos
    console.log('\n   📈 Indicadores Técnicos:');
    console.log('      • RSI:', lstmForecast.technical_indicators.rsi.current.toFixed(2),
        `(${lstmForecast.technical_indicators.rsi.interpretation})`);
    console.log('      • MACD Signal:', lstmForecast.technical_indicators.macd.signal_cross);
    console.log('      • Bollinger Position:', 
        (lstmForecast.technical_indicators.bollinger_bands.price_position * 100).toFixed(0) + '%');
    
    // Trend
    console.log('\n   🔄 Trend Analysis:');
    console.log('      • Trend:', lstmForecast.trend_analysis.overall_trend);
    console.log('      • Strength:', (lstmForecast.trend_analysis.trend_strength * 100).toFixed(0) + '%');
    console.log('      • Change:', lstmForecast.trend_analysis.change_percent.toFixed(2) + '%');
    
    // Stat
    console.log('\n   📊 Estatísticas:');
    console.log('      • Mean:', lstmForecast.statistics.mean.toFixed(2));
    console.log('      • Std Dev:', lstmForecast.statistics.std.toFixed(4));
    console.log('      • Min/Max:', lstmForecast.statistics.min.toFixed(2), 
        '/', lstmForecast.statistics.max.toFixed(2));
    
    // Signals
    console.log('\n   🎯 Trading Signals:');
    console.log('      • Overall Signal:', lstmForecast.signals.overall_signal);
    console.log('      • Recommendation:', lstmForecast.signals.recommendation);
    console.log('      • Confidence:', (lstmForecast.signals.confidence * 100).toFixed(0) + '%');
    
    // Performance
    console.log('\n   ⭐ Performance:');
    console.log('      • Model Confidence:', 
        (lstmForecast.performance_summary.model_confidence * 100).toFixed(0) + '%');
    console.log('      • Prediction Reliability:', 
        lstmForecast.performance_summary.prediction_reliability);
    console.log('      • Risk Level:', lstmForecast.performance_summary.risk_level);

    // 6. Fazer Previsão Prophet
    console.log('\n6️⃣  Prophet Forecast (with COMPLETE data)...');
    const prophetForecast = await api.forecastProphet(prophetModelId, 10);
    
    console.log('   ✅ Prophet Forecast generated');
    console.log('      • Forecast Components:', Object.keys(prophetForecast.forecast_components).length);
    console.log('      • Execution time:', prophetForecast.execution_time_ms.toFixed(2), 'ms');

  } catch (error) {
    console.error('❌ Error:', error.response?.data || error.message);
  }
}

// ============================================================
// 3. EXEMPLO - EXTRAIR E USAR DADOS PARA GRÁFICOS
// ============================================================

async function exampleExtractDataForCharts() {
  console.log('\n' + '='.repeat(60));
  console.log('📊 EXEMPLO - Extrair Dados para Gráficos');
  console.log('='.repeat(60));

  const api = new TOTEMDeepseaClient(
    process.env.API_KEY,
    process.env.API_HOST || 'http://localhost:8000'
  );

  try {
    // Listar modelos
    const models = await api.getModels();
    if (models.total === 0) {
      console.log('❌ Nenhum modelo encontrado. Execute o exemplo 1 primeiro!');
      return;
    }

    const modelId = Object.keys(models.models)[0];
    console.log(`\nUsando modelo: ${modelId}`);

    // Obter forecast
    const forecast = await api.forecastLSTM(modelId, 24);

    // ================================================
    // Extrair dados para Chart.js
    // ================================================
    console.log('\n📈 Dados para Chart.js:');
    const chartData = {
      labels: forecast.timestamps.dates,
      datasets: [
        {
          label: 'Forecast',
          data: forecast.forecast.values.map(row => row[0]),
          borderColor: 'rgb(75, 192, 192)',
          tension: 0.1,
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
      ],
    };
    console.log('   ✅ Chart data:', JSON.stringify(chartData, null, 2).slice(0, 200) + '...');

    // ================================================
    // Extrair dados para Recharts (React)
    // ================================================
    console.log('\n📊 Dados para Recharts (React):');
    const rechartsData = forecast.timestamps.dates.map((date, idx) => ({
      date,
      forecast: forecast.forecast.values[idx] ? forecast.forecast.values[idx][0] : null,
      upper95: forecast.confidence_intervals.upper_bound_95[idx] 
        ? forecast.confidence_intervals.upper_bound_95[idx][0] : null,
      lower95: forecast.confidence_intervals.lower_bound_95[idx] 
        ? forecast.confidence_intervals.lower_bound_95[idx][0] : null,
      rsi: forecast.technical_indicators.rsi.values[idx] || null,
      macd: forecast.technical_indicators.macd.macd_line[idx] || null,
    }));
    console.log('   ✅ Recharts data (first 3 rows):', 
      JSON.stringify(rechartsData.slice(0, 3), null, 2));

    // ================================================
    // Extrair Indicadores para Dashboard
    // ================================================
    console.log('\n🎯 Indicadores para Dashboard:');
    const dashboard = {
      // KPIs
      kpis: {
        modelConfidence: (forecast.performance_summary.model_confidence * 100).toFixed(1) + '%',
        riskLevel: forecast.performance_summary.risk_level.toUpperCase(),
        recommendation: forecast.performance_summary.recommendation,
        forecastTrend: forecast.trend_analysis.overall_trend.toUpperCase(),
      },
      
      // Técnicos
      technicals: {
        rsi: {
          value: forecast.technical_indicators.rsi.current.toFixed(2),
          status: forecast.technical_indicators.rsi.interpretation,
        },
        macd: {
          signal: forecast.technical_indicators.macd.signal_cross,
          current: forecast.technical_indicators.macd.current_macd.toFixed(4),
        },
        bollinger: {
          position: (forecast.technical_indicators.bollinger_bands.price_position * 100).toFixed(0),
          bandwidth: forecast.technical_indicators.bollinger_bands.band_width.toFixed(4),
        },
      },
      
      // Metricas
      metrics: {
        volatility: (forecast.trend_analysis.volatility * 100).toFixed(2) + '%',
        mean: forecast.statistics.mean.toFixed(2),
        maxVariance: (forecast.statistics.max - forecast.statistics.min).toFixed(2),
      },
      
      // Signals
      signals: {
        buy: forecast.signals.buy_signals,
        sell: forecast.signals.sell_signals,
        confidence: (forecast.signals.confidence * 100).toFixed(0) + '%',
      }
    };
    console.log('   ✅ Dashboard KPIs:', JSON.stringify(dashboard, null, 2));

  } catch (error) {
    console.error('❌ Error:', error.message);
  }
}

// ============================================================
// 4. EXEMPLO - MONITORAMENTO EM TEMPO REAL
// ============================================================

async function exampleRealTimeMonitoring() {
  console.log('\n' + '='.repeat(60));
  console.log('🔴 EXEMPLO - Monitoramento em Tempo Real');
  console.log('='.repeat(60));

  const api = new TOTEMDeepseaClient(
    process.env.API_KEY,
    process.env.API_HOST || 'http://localhost:8000'
  );

  try {
    const models = await api.getModels();
    if (models.total === 0) {
      console.log('❌ Nenhum modelo encontrado!');
      return;
    }

    const modelId = Object.keys(models.models)[0];

    // Simular monitoramento
    console.log(`\nMonitorando ${modelId} a cada 2 segundos (5 iterações)...`);
    
    for (let i = 0; i < 5; i++) {
      console.log(`\n⏱️  Iteração ${i + 1}`);
      
      try {
        const forecast = await api.forecastLSTM(modelId, 5);
        
        console.log(`   • Latest forecast: ${forecast.forecast.values[forecast.forecast.values.length - 1][0].toFixed(2)}`);
        console.log(`   • RSI: ${forecast.technical_indicators.rsi.current.toFixed(2)}`);
        console.log(`   • MACD Signal: ${forecast.technical_indicators.macd.signal_cross}`);
        console.log(`   • Recommendation: ${forecast.signals.recommendation}`);
        console.log(`   • Confidence: ${(forecast.signals.confidence * 100).toFixed(0)}%`);
        
        // Simular delay
        await new Promise(resolve => setTimeout(resolve, 1000));
      } catch (err) {
        console.error('   ❌ Error fetching forecast');
      }
    }

    console.log('\n✅ Monitoramento concluído!');

  } catch (error) {
    console.error('❌ Error:', error.message);
  }
}

// ============================================================
// EXECUTAR EXEMPLOS
// ============================================================

async function main() {
  try {
    // Exemplo 1: Workflow Completo
    await exampleCompleteWorkflow();
    
    // Exemplo 2: Extrair para Gráficos
    console.log('\n\n');
    await exampleExtractDataForCharts();
    
    // Exemplo 3: Monitoramento em Tempo Real
    console.log('\n\n');
    await exampleRealTimeMonitoring();

  } catch (error) {
    console.error('Fatal error:', error);
  }
}

// Executar se for diretamente
if (require.main === module) {
  main();
}

module.exports = { TOTEMDeepseaClient };

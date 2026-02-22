"""
TOTEM_DEEPSEA FastAPI
API para previsão de séries temporais multivariadas usando LSTM e Prophet
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
import numpy as np
import pandas as pd
import io
import os
import json
import logging
from datetime import datetime, timedelta
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.data_preprocessing import DataPreprocessor
from src.models.lstm_model import build_lstm_model, train_lstm
from src.models.prophet_model import train_prophet, forecast_prophet
from src.evaluation import calculate_metrics
from src.config import MODELS_PATH, RANDOM_SEED

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================
# CONFIGURAÇÕES
# ============================================================
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ============================================================
# MODELOS PYDANTIC
# ============================================================

class DataPoint(BaseModel):
    """Ponto único de dados de previsão"""
    timestamp: Optional[str] = None
    value: float


class ForecastMetrics(BaseModel):
    """Métricas de avaliação do modelo"""
    mae: Optional[float] = None
    rmse: Optional[float] = None
    mape: Optional[float] = None
    r2: Optional[float] = None
    model_type: str


class ForecastResponse(BaseModel):
    """Resposta padrão de previsão"""
    forecast: List[float] = Field(description="Valores previstos")
    actual: Optional[List[float]] = Field(None, description="Valores reais (se disponíveis)")
    timestamps: List[str] = Field(description="Timestamps dos pontos previstos")
    metrics: ForecastMetrics = Field(description="Métricas de desempenho")
    message: str = Field(description="Mensagem de status")
    success: bool = Field(description="Status de sucesso da requisição")


class TrainingRequest(BaseModel):
    """Requisição de treinamento"""
    lookback: int = Field(60, description="Janela de lookback para LSTM")
    epochs: int = Field(50, description="Número de epochs para treinamento")
    batch_size: int = Field(16, description="Tamanho do batch")
    test_size: float = Field(0.2, description="Proporção do conjunto de teste")


class ForecastRequest(BaseModel):
    """Requisição de previsão"""
    periods: int = Field(24, description="Número de períodos a prever")
    model_name: Optional[str] = Field(None, description="Nome do modelo a usar")


# ============================================================
# GERENCIADOR DE DADOS
# ============================================================

class DataManager:
    """Gerencia dados de upload e aramazenamento"""
    
    def __init__(self):
        self.uploaded_data: Dict[str, pd.DataFrame] = {}
        self.trained_models: Dict[str, Any] = {}
        self.scalers: Dict[str, Any] = {}
        logger.info("✓ DataManager inicializado")
    
    def save_upload(self, file: UploadFile, filename: str) -> pd.DataFrame:
        """Salvar e validar CSV enviado"""
        try:
            contents = file.file.read()
            df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
            
            # Validações básicas
            if df.empty:
                raise ValueError("CSV vazio")
            
            if df.shape[1] < 2:
                raise ValueError("CSV deve ter no mínimo 2 colunas (datetime + valor)")
            
            # Detectar coluna datetime
            datetime_col = None
            numeric_cols = []
            
            for col in df.columns:
                try:
                    pd.to_datetime(df[col])
                    datetime_col = col
                    break
                except:
                    if pd.api.types.is_numeric_dtype(df[col]):
                        numeric_cols.append(col)
            
            if datetime_col is None:
                logger.warning("Nenhuma coluna datetime detectada, usando índice")
            
            if not numeric_cols:
                raise ValueError("CSV deve conter no mínimo 1 coluna numérica")
            
            # Armazenar
            self.uploaded_data[filename] = df
            logger.info(f"✓ Arquivo salvo: {filename} ({df.shape[0]}x{df.shape[1]})")
            
            return df
        
        except Exception as e:
            logger.error(f"❌ Erro ao processar CSV: {e}")
            raise
    
    def get_data(self, filename: str) -> pd.DataFrame:
        """Recuperar dados enviados"""
        if filename not in self.uploaded_data:
            raise HTTPException(status_code=404, detail=f"Arquivo '{filename}' não encontrado")
        return self.uploaded_data[filename]
    
    def save_model(self, model_name: str, model: Any, model_type: str):
        """Armazenar modelo treinado"""
        self.trained_models[f"{model_name}_{model_type}"] = model
        logger.info(f"✓ Modelo salvo: {model_name}_{model_type}")
    
    def get_model(self, model_name: str, model_type: str) -> Any:
        """Recuperar modelo"""
        key = f"{model_name}_{model_type}"
        if key not in self.trained_models:
            raise HTTPException(status_code=404, detail=f"Modelo '{key}' não encontrado")
        return self.trained_models[key]


# ============================================================
# INICIALIZAR APLICAÇÃO E GERENCIADOR
# ============================================================

app = FastAPI(
    title="TOTEM_DEEPSEA API",
    description="Sistema de previsão de séries temporais multivariadas",
    version="1.0.0"
)

data_manager = DataManager()
preprocessor = DataPreprocessor()

# ============================================================
# FUNÇÕES AUXILIARES
# ============================================================

def extract_numeric_features(df: pd.DataFrame) -> tuple:
    """
    Extrair features numéricas do DataFrame
    
    Args:
        df: DataFrame com dados mistos
    
    Returns:
        tuple: (dados_numericos, nomes_colunas, coluna_datetime)
    """
    # Detectar coluna datetime
    datetime_col = None
    numeric_cols = []
    
    for col in df.columns:
        if col.lower() in ['date', 'datetime', 'timestamp', 'time']:
            datetime_col = col
        elif pd.api.types.is_numeric_dtype(df[col]):
            numeric_cols.append(col)
    
    if not numeric_cols:
        raise ValueError("Nenhuma coluna numérica encontrada")
    
    data = df[numeric_cols].values
    
    return data, numeric_cols, datetime_col


def prepare_forecast_response(
    forecast_values: np.ndarray,
    actual_values: Optional[np.ndarray] = None,
    timestamps: Optional[List[str]] = None,
    metrics_dict: Optional[Dict] = None,
    model_type: str = "LSTM"
) -> ForecastResponse:
    """
    Montar resposta JSON de previsão
    
    Args:
        forecast_values: Valores previstos
        actual_values: Valores reais (opcional)
        timestamps: Lista de timestamps
        metrics_dict: Dicionário de métricas
        model_type: Tipo de modelo usado
    
    Returns:
        ForecastResponse: Resposta estruturada
    """
    # Garantir lists
    forecast_list = forecast_values.flatten().tolist() if isinstance(forecast_values, np.ndarray) else forecast_values
    actual_list = actual_values.flatten().tolist() if actual_values is not None else None
    
    # Gerar timestamps se não fornecido
    if timestamps is None:
        now = datetime.now()
        timestamps = [(now + timedelta(hours=i)).isoformat() for i in range(len(forecast_list))]
    
    # Métricas
    if metrics_dict is None:
        metrics_dict = {}
    
    forecast_metrics = ForecastMetrics(
        mae=metrics_dict.get('mae'),
        rmse=metrics_dict.get('rmse'),
        mape=metrics_dict.get('mape'),
        r2=metrics_dict.get('r2'),
        model_type=model_type
    )
    
    return ForecastResponse(
        forecast=forecast_list,
        actual=actual_list,
        timestamps=timestamps,
        metrics=forecast_metrics,
        message=f"Previsão gerada com sucesso usando {model_type}",
        success=True
    )


# ============================================================
# ENDPOINTS DA API
# ============================================================

@app.get("/", tags=["Health"])
async def root():
    """Endpoint raiz - healthcheck"""
    return {
        "name": "🔮 TOTEM_DEEPSEA API",
        "status": "✅ online",
        "version": "1.0.0",
        "models_loaded": len(data_manager.trained_models),
        "files_uploaded": len(data_manager.uploaded_data)
    }


@app.post("/upload_csv", tags=["Data Management"])
async def upload_csv(file: UploadFile = File(...)):
    """
    Upload arquivo CSV com séries temporais
    
    - **file**: Arquivo CSV com colunas de datetime + colunas numéricas
    
    Retorna informações sobre o arquivo processado
    """
    try:
        # Validar tipo de arquivo
        if not file.filename.endswith('.csv'):
            raise HTTPException(status_code=400, detail="Apenas arquivos .csv são aceitos")
        
        # Processar arquivo
        df = data_manager.save_upload(file, file.filename)
        
        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "message": f"Arquivo '{file.filename}' enviado com sucesso",
                "filename": file.filename,
                "shape": {"rows": df.shape[0], "columns": df.shape[1]},
                "columns": list(df.columns),
                "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()}
            }
        )
    
    except Exception as e:
        logger.error(f"❌ Erro no upload: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/train_lstm", tags=["Model Training"])
async def train_lstm_model(
    filename: str,
    request: TrainingRequest = TrainingRequest()
):
    """
    Treinar modelo LSTM com dados do arquivo enviado
    
    - **filename**: Nome do arquivo CSV previamente enviado
    - **lookback**: Janela de lookback (padrão: 60)
    - **epochs**: Número de epochs (padrão: 50)
    - **batch_size**: Tamanho do batch (padrão: 16)
    - **test_size**: Proporção de teste (padrão: 0.2)
    """
    try:
        logger.info(f"🚀 Iniciando treinamento LSTM para {filename}...")
        
        # Recuperar dados
        df = data_manager.get_data(filename)
        
        # Extrair features numéricas
        data, numeric_cols, datetime_col = extract_numeric_features(df)
        
        # Pré-processamento
        logger.info(f"📊 Pré-processando {len(numeric_cols)} colunas")
        X_norm, _, _ = preprocessor.normalize_data(data, None, None)
        
        # Criar sequências
        X_seq, y_seq = preprocessor.create_sequences(X_norm, lookback=request.lookback)
        
        # Split
        X_train, X_val, X_test, y_train, y_val, y_test = \
            preprocessor.train_test_split(X_seq, y_seq, test_size=request.test_size)
        
        # Treinar modelo
        logger.info(f"🏗️  Construindo modelo LSTM com input shape {X_train.shape[1:]}")
        model, history = train_lstm(
            X_train, y_train, X_val, y_val,
            model_name=filename.replace('.csv', '')
        )
        
        # Armazenar modelo
        data_manager.save_model(
            filename.replace('.csv', ''),
            model,
            'lstm'
        )
        
        # Calcular métricas na validação
        y_pred = model.predict(X_test, verbose=0)
        metrics = calculate_metrics(y_test, y_pred)
        
        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "message": f"Modelo LSTM treinado com sucesso",
                "model_name": filename.replace('.csv', '') + "_lstm",
                "training_info": {
                    "epochs": request.epochs,
                    "lookback": request.lookback,
                    "batch_size": request.batch_size,
                    "final_train_loss": float(history.history['loss'][-1]),
                    "final_val_loss": float(history.history['val_loss'][-1]),
                },
                "test_metrics": {
                    "mae": float(metrics.get('mae', 0)),
                    "rmse": float(metrics.get('rmse', 0)),
                    "mape": float(metrics.get('mape', 0)),
                    "r2": float(metrics.get('r2', 0))
                },
                "data_shapes": {
                    "train": list(X_train.shape),
                    "val": list(X_val.shape),
                    "test": list(X_test.shape)
                }
            }
        )
    
    except Exception as e:
        logger.error(f"❌ Erro no treinamento LSTM: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/train_prophet", tags=["Model Training"])
async def train_prophet_model(
    filename: str,
    column_to_forecast: str = "Close"
):
    """
    Treinar modelo Prophet com dados do arquivo enviado
    
    - **filename**: Nome do arquivo CSV previamente enviado
    - **column_to_forecast**: Coluna numérica a prever (padrão: Close)
    """
    try:
        logger.info(f"🚀 Iniciando treinamento Prophet para {filename}...")
        
        # Recuperar dados
        df = data_manager.get_data(filename)
        
        # Validar coluna
        if column_to_forecast not in df.columns:
            raise ValueError(f"Coluna '{column_to_forecast}' não encontrada. Colunas disponíveis: {list(df.columns)}")
        
        # Preparar dados (Prophet requer 'ds' e 'y')
        df_prophet = pd.DataFrame({
            'ds': df.index if isinstance(df.index, pd.DatetimeIndex) else pd.date_range(start='2024-01-01', periods=len(df)),
            'y': df[column_to_forecast].values
        })
        
        # Treinar Prophet
        logger.info(f"🔮 Treinando Prophet...")
        model = train_prophet(
            df_prophet,
            col_target='y',
            col_datetime='ds',
            model_name=filename.replace('.csv', '')
        )
        
        # Armazenar modelo
        data_manager.save_model(
            filename.replace('.csv', ''),
            model,
            'prophet'
        )
        
        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "message": f"Modelo Prophet treinado com sucesso",
                "model_name": filename.replace('.csv', '') + "_prophet",
                "column_forecasted": column_to_forecast,
                "data_points": len(df_prophet),
                "message_details": "Modelo Prophet requer ao menos 2 anos de dados para sazonalidade anual"
            }
        )
    
    except Exception as e:
        logger.error(f"❌ Erro no treinamento Prophet: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/forecast_lstm", tags=["Forecasting"])
async def forecast_lstm(
    filename: str,
    periods: int = 24,
    model_name: Optional[str] = None
):
    """
    Gerar previsões com modelo LSTM
    
    - **filename**: Nome do arquivo CSV original
    - **periods**: Número de períodos a prever (padrão: 24)
    - **model_name**: Nome customizado do modelo (opcional)
    """
    try:
        logger.info(f"🔮 Gerando previsão LSTM para {periods} períodos...")
        
        # Recuperar dados e modelo
        df = data_manager.get_data(filename)
        model_key = model_name or filename.replace('.csv', '')
        model = data_manager.get_model(model_key, 'lstm')
        
        # Extrair features
        data, numeric_cols, datetime_col = extract_numeric_features(df)
        
        # Normalizar últimos dados
        X_norm, _, _ = preprocessor.normalize_data(data, None, None)
        
        # Usar últimos dados como base para previsão
        lookback = 60
        X_base = X_norm[-lookback:].reshape(1, lookback, -1)
        
        # Fazer previsões iterativas
        predictions = []
        current_input = X_base.copy()
        
        for _ in range(periods):
            pred = model.predict(current_input, verbose=0)
            predictions.append(pred[0, 0])  # Primeira feature
            
            # Atualizar input (remover primeira linha, adicionar previsão)
            current_input = np.concatenate([
                current_input[0, 1:, :],
                pred.reshape(1, 1, -1)
            ], axis=1).reshape(1, lookback, -1)
        
        # Gerar timestamps
        if datetime_col:
            last_date = pd.to_datetime(df[datetime_col].iloc[-1])
            timestamps = [(last_date + timedelta(hours=i)).isoformat() for i in range(1, periods + 1)]
        else:
            timestamps = [(datetime.now() + timedelta(hours=i)).isoformat() for i in range(1, periods + 1)]
        
        # Montar resposta
        response = prepare_forecast_response(
            forecast_values=np.array(predictions),
            actual_values=data[-periods:, 0] if len(data) >= periods else None,
            timestamps=timestamps,
            model_type="LSTM"
        )
        
        return response.dict()
    
    except Exception as e:
        logger.error(f"❌ Erro na previsão LSTM: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/forecast_prophet", tags=["Forecasting"])
async def forecast_prophet(
    filename: str,
    periods: int = 24,
    model_name: Optional[str] = None
):
    """
    Gerar previsões com modelo Prophet
    
    - **filename**: Nome do arquivo CSV original
    - **periods**: Número de períodos a prever (padrão: 24)
    - **model_name**: Nome customizado do modelo (opcional)
    """
    try:
        logger.info(f"🔮 Gerando previsão Prophet para {periods} períodos...")
        
        # Recuperar modelo
        model_key = model_name or filename.replace('.csv', '')
        model = data_manager.get_model(model_key, 'prophet')
        
        # Fazer previsão
        forecast = forecast_prophet(model, periods=periods)
        
        # Extrair valores
        forecast_values = forecast['yhat'].values
        timestamps = forecast['ds'].astype(str).values.tolist()
        
        # Recuperar valores reais do arquivo (se disponíveis)
        df = data_manager.get_data(filename)
        actual_values = df.iloc[-periods:].values.flatten() if len(df) >= periods else None
        
        # Montar resposta
        response = prepare_forecast_response(
            forecast_values=forecast_values,
            actual_values=actual_values,
            timestamps=timestamps,
            model_type="Prophet"
        )
        
        return response.dict()
    
    except Exception as e:
        logger.error(f"❌ Erro na previsão Prophet: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/models", tags=["Model Management"])
async def list_models():
    """Listar todos os modelos treinados"""
    models_info = []
    
    for model_name in data_manager.trained_models.keys():
        models_info.append({
            "name": model_name,
            "type": model_name.split('_')[-1]
        })
    
    return JSONResponse(
        status_code=200,
        content={
            "success": True,
            "total_models": len(models_info),
            "models": models_info
        }
    )


@app.get("/uploads", tags=["Model Management"])
async def list_uploads():
    """Listar todos os arquivos enviados"""
    uploads_info = []
    
    for filename, df in data_manager.uploaded_data.items():
        uploads_info.append({
            "filename": filename,
            "rows": df.shape[0],
            "columns": df.shape[1],
            "column_names": list(df.columns)
        })
    
    return JSONResponse(
        status_code=200,
        content={
            "success": True,
            "total_uploads": len(uploads_info),
            "uploads": uploads_info
        }
    )


@app.get("/health", tags=["Health"])
async def health_check():
    """Verificar saúde da API"""
    return {
        "status": "✅ Healthy",
        "timestamp": datetime.now().isoformat(),
        "models_in_memory": len(data_manager.trained_models),
        "dataframes_in_memory": len(data_manager.uploaded_data)
    }


# ============================================================
# INICIALIZAÇÃO
# ============================================================

if __name__ == "__main__":
    import uvicorn
    
    logger.info("🚀 Iniciando TOTEM_DEEPSEA API...")
    logger.info("📚 Documentação disponível em: http://localhost:8000/docs")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )

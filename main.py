import os
import logging
from typing import Optional, List
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from io import StringIO

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Query, Header, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
from dotenv import load_dotenv

load_dotenv()

from src.api_models import (
    UploadResponseSchema,
    TrainingRequestSchema,
    TrainingResponseSchema,
    ForecastRequestSchema,
    ForecastResponseSchema,
    MetricsSchema,
    ErrorResponseSchema,
    HealthCheckSchema
)
from src.api_utils import (
    FileManager,
    ModelManager,
    DataPreprocessor,
    MetricsCalculator,
    APIException,
    file_manager,
    model_manager
)
from src.auth import api_key_manager
from src.data_preprocessing import DataPreprocessor as DP
from src.models.lstm_model import build_lstm_model, train_lstm
from src.models.prophet_model import train_prophet
from src.technical_analysis import TechnicalAnalysisEngine, generate_signals
from src.config import (
    LSTM_LOOKBACK,
    LSTM_EPOCHS,
    LSTM_BATCH_SIZE,
    LSTM_LEARNING_RATE
)
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, r2_score
from prophet import Prophet


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="TOTEM_DEEPSEA API",
    description="Multi-target Time Series Forecasting System",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

active_files = {}
active_models = {}


async def verify_api_key(authorization: Optional[str] = Header(None)) -> Optional[dict]:
    if not authorization:
        return None
    
    # Try to extract key from "Bearer" format
    if authorization.startswith("Bearer "):
        api_key = authorization[7:]
    else:
        api_key = authorization
    
    # Validate key
    is_valid, metadata = api_key_manager.validate_key(api_key)
    
    if not is_valid:
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing API key"
        )
    
    return metadata


@app.on_event("startup")
async def startup_event():
    logger.info("🚀 TOTEM_DEEPSEA API starting...")
    logger.info(f"✓ Storage directory: {file_manager.storage_dir}")
    logger.info(f"✓ Models directory: {model_manager.models_dir}")


@app.exception_handler(APIException)
async def api_exception_handler(request, exc: APIException):
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponseSchema(
            error="API Error",
            message=exc.message,
            error_code=exc.error_code,
            timestamp=datetime.now().isoformat()
        ).dict()
    )


@app.get("/health", response_model=HealthCheckSchema)
async def health_check():
    return HealthCheckSchema(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        version="1.0.0"
    )


@app.post("/upload_csv", response_model=UploadResponseSchema)
async def upload_csv(
    file: UploadFile = File(...),
    key_data: dict = Depends(verify_api_key)
):
    try:
        contents = await file.read()
        csv_string = contents.decode('utf-8')
        
        df = pd.read_csv(StringIO(csv_string))
        logger.info(f"✓ CSV uploaded: {file.filename} ({len(df)} rows)")
        
        is_valid, message, datetime_col, numeric_cols = DataPreprocessor.validate_csv(df)
        
        if not is_valid:
            logger.error(f"❌ CSV validation failed: {message}")
            raise APIException(
                message=message,
                error_code="INVALID_CSV",
                status_code=400
            )
        
        file_id = file_manager.save_csv(df)
        
        active_files[file_id] = {
            'data': df,
            'columns': df.columns.tolist(),
            'datetime_col': datetime_col,
            'numeric_cols': numeric_cols,
            'rows': len(df),
            'created_at': datetime.now().isoformat()
        }
        
        logger.info(f"✓ File registered: {file_id}")
        
        return UploadResponseSchema(
            file_id=file_id,
            rows=len(df),
            columns=df.columns.tolist(),
            datetime_column=datetime_col,
            numeric_columns=numeric_cols
        )
    
    except APIException:
        raise
    except Exception as e:
        logger.error(f"❌ Upload error: {str(e)}")
        raise APIException(
            message=f"Failed to process CSV: {str(e)}",
            error_code="UPLOAD_ERROR",
            status_code=400
        )


@app.post("/train_lstm", response_model=TrainingResponseSchema)
async def train_lstm_endpoint(
    request: TrainingRequestSchema,
    key_data: dict = Depends(verify_api_key)
):
    try:
        file_id = request.file_id
        lookback = request.lookback or LSTM_LOOKBACK
        epochs = request.epochs or LSTM_EPOCHS
        batch_size = request.batch_size or LSTM_BATCH_SIZE
        
        if file_id not in active_files:
            raise APIException(
                message=f"File not found: {file_id}",
                error_code="FILE_NOT_FOUND",
                status_code=404
            )
        
        file_data = active_files[file_id]
        df = file_data['data']
        numeric_cols = file_data['numeric_cols']
        
        logger.info(f"🔄 Training LSTM on {file_id}...")
        start_time = datetime.now()
        
        data = df[numeric_cols].values.astype(np.float32)
        data = np.nan_to_num(data, nan=0.0, posinf=1e6, neginf=-1e6)
        
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(data)
        
        X, y = [], []
        for i in range(len(X_scaled) - lookback):
            X.append(X_scaled[i:(i + lookback)])
            y.append(X_scaled[i + lookback])
        
        X = np.array(X)
        y = np.array(y)
        
        if len(X) == 0:
            raise APIException(
                message="Insufficient data after preprocessing",
                error_code="INSUFFICIENT_DATA",
                status_code=400
            )
        
        train_size = int(len(X) * 0.8)
        X_train, X_val = X[:train_size], X[train_size:]
        y_train, y_val = y[:train_size], y[train_size:]
        
        model, history = train_lstm(X_train, y_train, X_val, y_val, model_name=f'lstm_{file_id}')
        
        train_loss = float(history.history['loss'][-1])
        
        model_id = f"lstm_{file_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"[:32]
        
        metadata = {
            'type': 'lstm',
            'file_id': file_id,
            'lookback': lookback,
            'epochs': epochs,
            'batch_size': batch_size,
            'numeric_cols': numeric_cols,
            'train_loss': train_loss,
            'created_at': datetime.now().isoformat()
        }
        
        model_manager.save_model(model, model_id, metadata)
        
        scaler_id = f"{model_id}_scaler"
        model_manager.save_model(scaler, scaler_id)
        
        training_time = (datetime.now() - start_time).total_seconds()
        
        logger.info(f"✓ LSTM model trained: {model_id} ({training_time:.2f}s)")
        
        active_models[model_id] = metadata
        
        return TrainingResponseSchema(
            model_id=model_id,
            model_type="lstm",
            metrics=MetricsSchema(
                mae=train_loss,
                rmse=train_loss,
                mape=None,
                r2=None
            ),
            training_time=training_time
        )
    
    except APIException:
        raise
    except Exception as e:
        logger.error(f"❌ LSTM training error: {str(e)}")
        raise APIException(
            message=f"LSTM training failed: {str(e)}",
            error_code="TRAINING_ERROR",
            status_code=500
        )


@app.post("/train_prophet", response_model=TrainingResponseSchema)
async def train_prophet_endpoint(
    request: TrainingRequestSchema,
    key_data: dict = Depends(verify_api_key)
):
    try:
        file_id = request.file_id
        
        if file_id not in active_files:
            raise APIException(
                message=f"File not found: {file_id}",
                error_code="FILE_NOT_FOUND",
                status_code=404
            )
        
        file_data = active_files[file_id]
        df = file_data['data']
        datetime_col = file_data['datetime_col']
        numeric_cols = file_data['numeric_cols']
        
        logger.info(f"🔄 Training Prophet on {file_id}...")
        start_time = datetime.now()
        
        models = {}
        
        for col in numeric_cols:
            prophet_df = pd.DataFrame({
                'ds': pd.to_datetime(df[datetime_col]) if datetime_col else pd.date_range(start='2020-01-01', periods=len(df)),
                'y': df[col].values
            })
            
            prophet_df = prophet_df.dropna()
            
            m = Prophet()
            m.fit(prophet_df)
            
            models[col] = m
        
        model_id = f"prophet_{file_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"[:32]
        
        metadata = {
            'type': 'prophet',
            'file_id': file_id,
            'numeric_cols': numeric_cols,
            'created_at': datetime.now().isoformat()
        }
        
        model_manager.save_model(models, model_id, metadata)
        
        training_time = (datetime.now() - start_time).total_seconds()
        
        logger.info(f"✓ Prophet models trained: {model_id} ({training_time:.2f}s)")
        
        active_models[model_id] = metadata
        
        return TrainingResponseSchema(
            model_id=model_id,
            model_type="prophet",
            metrics=MetricsSchema(mae=None, rmse=None, mape=None, r2=None),
            training_time=training_time
        )
    
    except APIException:
        raise
    except Exception as e:
        logger.error(f"❌ Prophet training error: {str(e)}")
        raise APIException(
            message=f"Prophet training failed: {str(e)}",
            error_code="TRAINING_ERROR",
            status_code=500
        )


@app.get("/forecast_lstm")
async def forecast_lstm_endpoint(
    model_id: str = Query(..., description="Trained LSTM model ID"),
    periods: int = Query(24, ge=1, le=365, description="Number of periods to forecast"),
    key_data: dict = Depends(verify_api_key)
):
    start_time = datetime.now()
    
    try:
        model, metadata = model_manager.load_model(model_id)
        if model is None:
            raise APIException(message=f"Model not found: {model_id}", error_code="MODEL_NOT_FOUND", status_code=404)
        if metadata is None:
            raise APIException(message=f"Model metadata not found: {model_id}", error_code="MODEL_NOT_FOUND", status_code=404)
        
        scaler_id = f"{model_id}_scaler"
        scaler, _ = model_manager.load_model(scaler_id)
        if scaler is None:
            raise APIException(message="Scaler not found", error_code="SCALER_NOT_FOUND", status_code=500)
        
        file_id = metadata['file_id']
        if file_id not in active_files:
            raise APIException(message="File not found", error_code="FILE_NOT_FOUND", status_code=404)
        
        file_data = active_files[file_id]
        df = file_data['data']
        datetime_col = file_data['datetime_col']
        numeric_cols = metadata['numeric_cols']
        lookback = metadata['lookback']
        
        logger.info(f"🔄 LSTM Forecast with FULL ANALYSIS: {model_id}")
        
        data = df[numeric_cols].values.astype(np.float32)
        data = np.nan_to_num(data, nan=0.0, posinf=1e6, neginf=-1e6)
        X_scaled = scaler.transform(data)
        
        last_sequence = X_scaled[-lookback:].reshape(1, lookback, len(numeric_cols))
        forecast_list = []
        std_errors = []
        current_sequence = last_sequence.copy()
        
        for _ in range(periods):
            next_pred = model.predict(current_sequence, verbose=0)
            forecast_list.append(next_pred[0])
            
            std_errors.append(np.std(next_pred))
            
            current_sequence = np.append(current_sequence[0, 1:, :], next_pred, axis=0).reshape(1, lookback, len(numeric_cols))
        
        forecast_array = np.array(forecast_list)
        forecast_inverse = scaler.inverse_transform(forecast_array)
        std_errors_array = np.array(std_errors)
        
        lower_95, upper_95 = TechnicalAnalysisEngine.calculate_confidence_intervals(
            forecast_inverse, std_errors_array, 0.95
        )
        lower_80, upper_80 = TechnicalAnalysisEngine.calculate_confidence_intervals(
            forecast_inverse, std_errors_array, 0.80
        )
        
        actual_data = df[numeric_cols].tail(periods).values.astype(np.float32) if len(df) >= periods else None
        
        metrics = None
        if actual_data is not None:
            mae = float(mean_absolute_error(actual_data, forecast_inverse))
            rmse = float(np.sqrt(mean_squared_error(actual_data, forecast_inverse)))
            mape = float(mean_absolute_percentage_error(actual_data, forecast_inverse))
            r2 = float(r2_score(actual_data, forecast_inverse))
            directional_acc = TechnicalAnalysisEngine.calculate_directional_accuracy(actual_data[:, 0], forecast_inverse[:, 0])
            
            metrics = {
                "mae": mae,
                "rmse": rmse,
                "mape": mape,
                "r2": r2,
                "directional_accuracy": directional_acc
            }
        
        forecast_main = forecast_inverse[:, 0]
        
        rsi_data = TechnicalAnalysisEngine.calculate_rsi(forecast_main, period=14)
        macd_data = TechnicalAnalysisEngine.calculate_macd(forecast_main)
        bb_data = TechnicalAnalysisEngine.calculate_bollinger_bands(forecast_main, period=20)
        ma_data = TechnicalAnalysisEngine.calculate_moving_averages(forecast_main, [10, 20, 50])
        
        trend_data = TechnicalAnalysisEngine.calculate_trend_analysis(forecast_main)
        
        anomaly_data = TechnicalAnalysisEngine.detect_anomalies(forecast_main, threshold=2.5)
        
        stats_data = TechnicalAnalysisEngine.calculate_statistics(forecast_main)
        
        correlations = {}
        if actual_data is not None:
            correlations = TechnicalAnalysisEngine.calculate_correlations(
                forecast_main, 
                actual_data[:, 0],
                volume=actual_data[:, 1] if len(actual_data[0]) > 1 else None,
                rsi=rsi_data.get("values"),
                macd=macd_data.get("macd_line")
            )
        
        technical_indicators = {
            "rsi": rsi_data,
            "macd": macd_data,
            "bollinger_bands": bb_data,
            "moving_averages": ma_data
        }
        signals = generate_signals({"rsi": rsi_data, "macd": macd_data, "bollinger_bands": bb_data})
        
        timestamps_list = DataPreprocessor.generate_timestamps(df, datetime_col, periods)
        
        execution_time = (datetime.now() - start_time).total_seconds() * 1000
        
        response = {
            "model_id": model_id,
            "model_type": "lstm",
            "forecast_date": datetime.now().isoformat(),
            "periods": periods,
            
            "forecast": {
                "values": forecast_inverse.tolist(),
                "column_names": numeric_cols,
                "data_type": "float32"
            },
            
            "timestamps": {
                "dates": [t if isinstance(t, str) else t.isoformat() for t in timestamps_list],
                "unix_timestamps": [int(datetime.fromisoformat(t if isinstance(t, str) else t.isoformat()).timestamp()) if isinstance(t, (str, datetime)) else t for t in timestamps_list],
                "interval": "1h",
                "timezone": "UTC"
            },
            
            "confidence_intervals": {
                "lower_bound_95": lower_95.tolist(),
                "upper_bound_95": upper_95.tolist(),
                "lower_bound_80": lower_80.tolist(),
                "upper_bound_80": upper_80.tolist()
            },
            
            "actual_vs_forecast": {
                "actual_last_24": actual_data[:24].tolist() if actual_data is not None else None,
                "forecast_24": forecast_inverse[:24].tolist(),
                **(metrics if metrics else {})
            },
            
            "statistics": stats_data,
            
            "technical_indicators": technical_indicators,
            
            "trend_analysis": trend_data,
            
            "anomalies": anomaly_data,
            
            "correlation_analysis": correlations,
            
            "signals": signals,
            
            "performance_summary": {
                "model_confidence": float(metrics.get('r2', 0.0)) if metrics else 0.0,
                "prediction_reliability": "high" if (metrics and metrics.get('r2', 0) > 0.85) else ("medium" if (metrics and metrics.get('r2', 0) > 0.7) else "low"),
                "recommendation": signals.get("recommendation", "HOLD"),
                "risk_level": "low" if trend_data.get("volatility", 0) < 0.02 else ("high" if trend_data.get("volatility", 0) > 0.05 else "medium")
            },
            
            "generated_at": datetime.now().isoformat(),
            "execution_time_ms": execution_time,
            "cache_hit": False
        }
        
        logger.info(f"✓ LSTM forecast complete with full analysis: {execution_time:.2f}ms")
        
        return response
    
    except APIException:
        raise
    except Exception as e:
        logger.error(f"❌ LSTM forecast error: {str(e)}")
        raise APIException(
            message=f"LSTM forecast failed: {str(e)}",
            error_code="FORECAST_ERROR",
            status_code=500
        )


@app.get("/forecast_prophet")
async def forecast_prophet_endpoint(
    model_id: str = Query(..., description="Trained Prophet model ID"),
    periods: int = Query(24, ge=1, le=365, description="Number of periods to forecast"),
    key_data: dict = Depends(verify_api_key)
):
    start_time = datetime.now()
    
    try:
        models, metadata = model_manager.load_model(model_id)
        if models is None:
            raise APIException(message=f"Model not found: {model_id}", error_code="MODEL_NOT_FOUND", status_code=404)
        if metadata is None:
            raise APIException(message=f"Model metadata not found: {model_id}", error_code="MODEL_NOT_FOUND", status_code=404)
        
        file_id = metadata['file_id']
        if file_id not in active_files:
            raise APIException(message="File not found", error_code="FILE_NOT_FOUND", status_code=404)
        
        file_data = active_files[file_id]
        df = file_data['data']
        datetime_col = file_data['datetime_col']
        numeric_cols = metadata['numeric_cols']
        
        logger.info(f"🔄 Prophet Forecast with FULL ANALYSIS: {model_id}")
        
        forecast_all = []
        forecast_components = {}
        uncertainties = {}
        
        for col in numeric_cols:
            prophet_model = models[col]
            future = prophet_model.make_future_dataframe(periods=periods)
            forecast = prophet_model.predict(future)
            
            forecast_values = forecast['yhat'].values[-periods:]
            forecast_all.append(forecast_values)
            
            if 'trend' in forecast.columns:
                forecast_components[f"{col}_trend"] = forecast['trend'].values[-periods:].tolist()
            
            yearly_cols = [c for c in forecast.columns if 'yearly' in c]
            weekly_cols = [c for c in forecast.columns if 'weekly' in c]
            
            if yearly_cols:
                forecast_components[f"{col}_yearly"] = forecast[yearly_cols[0]].values[-periods:].tolist()
            if weekly_cols:
                forecast_components[f"{col}_weekly"] = forecast[weekly_cols[0]].values[-periods:].tolist()
            
            trend_unc = forecast['trend'].std() if 'trend' in forecast.columns else 0
            uncertainties[f"{col}_trend_uncertainty"] = [float(trend_unc)] * periods
        
        forecast_array = np.array(forecast_all).T
        
        actual_data = df[numeric_cols].tail(periods).values if len(df) >= periods else None
        
        metrics = None
        if actual_data is not None:
            mae = float(mean_absolute_error(actual_data, forecast_array))
            rmse = float(np.sqrt(mean_squared_error(actual_data, forecast_array)))
            mape = float(mean_absolute_percentage_error(actual_data, forecast_array))
            r2 = float(r2_score(actual_data, forecast_array))
            
            metrics = {
                "mae": mae,
                "rmse": rmse,
                "mape": mape,
                "r2": r2
            }
        
        forecast_main = forecast_array[:, 0]
        
        rsi_data = TechnicalAnalysisEngine.calculate_rsi(forecast_main, period=14)
        macd_data = TechnicalAnalysisEngine.calculate_macd(forecast_main)
        bb_data = TechnicalAnalysisEngine.calculate_bollinger_bands(forecast_main, period=20)
        ma_data = TechnicalAnalysisEngine.calculate_moving_averages(forecast_main, [10, 20, 50])
        
        trend_data = TechnicalAnalysisEngine.calculate_trend_analysis(forecast_main)
        
        anomaly_data = TechnicalAnalysisEngine.detect_anomalies(forecast_main, threshold=2.5)
        
        stats_data = TechnicalAnalysisEngine.calculate_statistics(forecast_main)
        
        technical_indicators = {
            "rsi": rsi_data,
            "macd": macd_data,
            "bollinger_bands": bb_data,
            "moving_averages": ma_data
        }
        signals = generate_signals({"rsi": rsi_data, "macd": macd_data, "bollinger_bands": bb_data})
        
        timestamps_list = DataPreprocessor.generate_timestamps(df, datetime_col, periods)
        
        execution_time = (datetime.now() - start_time).total_seconds() * 1000
        
        response = {
            "model_id": model_id,
            "model_type": "prophet",
            "forecast_date": datetime.now().isoformat(),
            "periods": periods,
            
            "forecast": {
                "values": forecast_array.tolist(),
                "column_names": numeric_cols
            },
            
            "timestamps": {
                "dates": [t if isinstance(t, str) else t.isoformat() for t in timestamps_list],
                "unix_timestamps": [int(datetime.fromisoformat(t if isinstance(t, str) else t.isoformat()).timestamp()) if isinstance(t, (str, datetime)) else t for t in timestamps_list],
                "interval": "1h",
                "timezone": "UTC"
            },
            
            "forecast_components": forecast_components,
            
            "uncertainties": {
                "trend_uncertainty": uncertainties.get(f"{numeric_cols[0]}_trend_uncertainty", [0.0] * periods),
                "observation_error": 0.02
            },
            
            "actual_vs_forecast": {
                "actual": actual_data[:24].tolist() if actual_data is not None else None,
                "forecast": forecast_array[:24].tolist(),
                **(metrics if metrics else {})
            },
            
            "statistics": stats_data,
            
            "technical_indicators": technical_indicators,
            
            "trend_analysis": trend_data,
            
            "anomalies": anomaly_data,
            
            "signals": signals,
            
            "performance_summary": {
                "model_confidence": float(metrics.get('r2', 0.0)) if metrics else 0.0,
                "prediction_reliability": "high" if (metrics and metrics.get('r2', 0) > 0.85) else ("medium" if (metrics and metrics.get('r2', 0) > 0.7) else "low"),
                "recommendation": signals.get("recommendation", "HOLD"),
                "risk_level": "low" if trend_data.get("volatility", 0) < 0.02 else ("high" if trend_data.get("volatility", 0) > 0.05 else "medium")
            },
            
            "generated_at": datetime.now().isoformat(),
            "execution_time_ms": execution_time,
            "cache_hit": False
        }
        
        logger.info(f"✓ Prophet forecast complete with full analysis: {execution_time:.2f}ms")
        
        return response
    
    except APIException:
        raise
    except Exception as e:
        logger.error(f"❌ Prophet forecast error: {str(e)}")
        raise APIException(
            message=f"Prophet forecast failed: {str(e)}",
            error_code="FORECAST_ERROR",
            status_code=500
        )


@app.get("/technical_analysis/{model_id}")
async def technical_analysis_endpoint(
    model_id: str,
    periods: int = Query(24, ge=1, le=365),
    key_data: dict = Depends(verify_api_key)
):
    try:
        forecast_resp = await forecast_lstm_endpoint(model_id, periods, key_data)
        
        analysis_response = {
            "model_id": model_id,
            "analysis_date": datetime.now().isoformat(),
            "indicators": forecast_resp.get("technical_indicators", {}),
            "signals": forecast_resp.get("signals", {}),
            "trend_analysis": forecast_resp.get("trend_analysis", {}),
            "anomalies": forecast_resp.get("anomalies", {}),
            "statistics": forecast_resp.get("statistics", {})
        }
        
        return analysis_response
    
    except APIException:
        raise
    except Exception as e:
        logger.error(f"❌ Technical analysis error: {str(e)}")
        raise APIException(
            message=f"Technical analysis failed: {str(e)}",
            error_code="ANALYSIS_ERROR",
            status_code=500
        )


@app.get("/models")
async def list_models():
    return {
        'total': len(active_models),
        'models': active_models
    }


@app.get("/files")
async def list_files():
    files_info = {}
    for file_id, file_data in active_files.items():
        files_info[file_id] = {
            'rows': file_data['rows'],
            'columns': file_data['columns'],
            'datetime_column': file_data['datetime_col'],
            'numeric_columns': file_data['numeric_cols'],
            'created_at': file_data['created_at']
        }
    
    return {
        'total': len(active_files),
        'files': files_info
    }


@app.delete("/cleanup/{file_id}")
async def cleanup_file(file_id: str):
    deleted_models = []
    
    for model_id in list(active_models.keys()):
        if active_models[model_id].get('file_id') == file_id:
            model_manager.load_model(model_id)
            deleted_models.append(model_id)
            del active_models[model_id]
    
    if file_id in active_files:
        file_manager.delete_file(file_id)
        del active_files[file_id]
        
        return {
            'status': 'success',
            'deleted_file': file_id,
            'deleted_models': deleted_models
        }
    
    raise APIException(
        message=f"File not found: {file_id}",
        error_code="FILE_NOT_FOUND",
        status_code=404
    )


# ============================================================
# API KEY MANAGEMENT ENDPOINTS
# ============================================================

@app.post("/generate-api-key")
async def generate_api_key(
    name: str = Query(..., description="Name/description for API key"),
    key_data: dict = Depends(verify_api_key)
):
    try:
        api_key = api_key_manager.generate_key(name, permissions=['*'])
        
        logger.info(f"✓ New API key generated: {name}")
        
        return {
            'status': 'success',
            'name': name,
            'api_key': api_key,
            'message': '⚠️  Store this key securely! You won\'t be able to retrieve it again.',
            'usage': f'Authorization: Bearer {api_key}'
        }
    
    except Exception as e:
        logger.error(f"❌ Error generating API key: {str(e)}")
        raise APIException(
            message=str(e),
            error_code="KEY_GENERATION_ERROR",
            status_code=500
        )


@app.get("/api-keys")
async def list_api_keys(key_data: dict = Depends(verify_api_key)):
    try:
        keys = api_key_manager.list_keys()
        
        return {
            'total': len(keys),
            'keys': keys
        }
    
    except Exception as e:
        logger.error(f"❌ Error listing keys: {str(e)}")
        raise APIException(
            message=str(e),
            error_code="LIST_ERROR",
            status_code=500
        )


@app.delete("/api-keys/{key_partial}")
async def revoke_api_key(
    key_partial: str = Query(..., description="First 16 chars of key hash"),
    key_data: dict = Depends(verify_api_key)
):
    try:
        success = api_key_manager.revoke_key(key_partial)
        
        if success:
            return {'status': 'success', 'message': 'API key revoked'}
        else:
            raise APIException(
                message="Key not found",
                error_code="KEY_NOT_FOUND",
                status_code=404
            )
    
    except Exception as e:
        logger.error(f"❌ Error revoking key: {str(e)}")
        raise APIException(
            message=str(e),
            error_code="REVOKE_ERROR",
            status_code=500
        )


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )

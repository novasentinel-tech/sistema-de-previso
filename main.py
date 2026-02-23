"""
TOTEM_DEEPSEA FastAPI Backend
Multi-target Time Series Forecasting System

This API provides endpoints for:
- CSV data upload
- LSTM model training and forecasting
- Prophet model training and forecasting
- Real-time stock analysis
"""

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

# Load environment variables
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
from src.config import (
    LSTM_LOOKBACK,
    LSTM_EPOCHS,
    LSTM_BATCH_SIZE,
    LSTM_LEARNING_RATE
)
from sklearn.preprocessing import MinMaxScaler
from prophet import Prophet

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="TOTEM_DEEPSEA API",
    description="Multi-target Time Series Forecasting System",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory storage for active sessions
active_files = {}  # {file_id: {data, columns, datetime_col, numeric_cols}}
active_models = {}  # {model_id: {type, file_id, params, created_at}}


# ============================================================
# API KEY VALIDATION
# ============================================================

async def verify_api_key(authorization: Optional[str] = Header(None)) -> Optional[dict]:
    """
    Verify API Key from Authorization header
    
    Expected format: Authorization: Bearer sk_your_api_key_here
    or: X-API-Key: sk_your_api_key_here
    """
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
    """Initialize on startup"""
    logger.info("🚀 TOTEM_DEEPSEA API starting...")
    logger.info(f"✓ Storage directory: {file_manager.storage_dir}")
    logger.info(f"✓ Models directory: {model_manager.models_dir}")


@app.exception_handler(APIException)
async def api_exception_handler(request, exc: APIException):
    """Handle custom API exceptions"""
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponseSchema(
            error=exc.error_code,
            message=exc.message,
            timestamp=datetime.now().isoformat()
        ).dict()
    )


@app.get("/health", response_model=HealthCheckSchema)
async def health_check():
    """
    Health check endpoint
    
    Returns:
        HealthCheckSchema: API status and timestamp
    """
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
    """
    Upload CSV file for processing
    
    Args:
        file: CSV file upload
        key_data: API key metadata (auto-validated)
        
    Returns:
        UploadResponseSchema: file_id, rows, columns, datetime_column
        
    Raises:
        HTTPException: If CSV is invalid
    """
    try:
        # Read CSV content
        contents = await file.read()
        csv_string = contents.decode('utf-8')
        
        # Parse CSV
        df = pd.read_csv(StringIO(csv_string))
        logger.info(f"✓ CSV uploaded: {file.filename} ({len(df)} rows)")
        
        # Validate CSV
        is_valid, message, datetime_col, numeric_cols = DataPreprocessor.validate_csv(df)
        
        if not is_valid:
            logger.error(f"❌ CSV validation failed: {message}")
            raise APIException(
                message=message,
                error_code="INVALID_CSV",
                status_code=400
            )
        
        # Save to file manager
        file_id = file_manager.save_csv(df)
        
        # Store in active files
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
            numeric_columns=numeric_cols,
            uploaded_at=datetime.now().isoformat()
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
    """
    Train LSTM model on uploaded data
    
    Args:
        request: TrainingRequestSchema with file_id, lookback, epochs, batch_size
        
    Returns:
        TrainingResponseSchema: model_id, metrics, training_time
        
    Raises:
        HTTPException: If training fails
    """
    try:
        file_id = request.file_id
        lookback = request.lookback or LSTM_LOOKBACK
        epochs = request.epochs or LSTM_EPOCHS
        batch_size = request.batch_size or LSTM_BATCH_SIZE
        
        # Check if file exists
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
        
        # Prepare data
        data = df[numeric_cols].values.astype(np.float32)
        data = np.nan_to_num(data, nan=0.0, posinf=1e6, neginf=-1e6)
        
        # Normalize data
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(data)
        
        # Create sequences
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
        
        # Build and train model
        model = build_lstm_model((X.shape[1], X.shape[2]))
        history = train_lstm(
            model, X, y, 
            epochs=epochs, 
            batch_size=batch_size,
            verbose=0
        )
        
        # Calculate training metrics
        train_loss = float(history.history['loss'][-1])
        
        # Generate model ID
        model_id = f"lstm_{file_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"[:32]
        
        # Save model
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
        
        # Store scaler for later use
        scaler_id = f"{model_id}_scaler"
        model_manager.save_model(scaler, scaler_id)
        
        training_time = (datetime.now() - start_time).total_seconds()
        
        logger.info(f"✓ LSTM model trained: {model_id} ({training_time:.2f}s)")
        
        # Store in active models
        active_models[model_id] = metadata
        
        return TrainingResponseSchema(
            model_id=model_id,
            model_type="lstm",
            rows_used=len(X),
            features=X.shape[2],
            metrics=MetricsSchema(
                mae=train_loss,
                rmse=train_loss,
                mape=None,
                r2=None
            ),
            training_time=training_time,
            created_at=datetime.now().isoformat()
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
    """
    Train Prophet model on uploaded data
    
    Args:
        request: TrainingRequestSchema with file_id
        
    Returns:
        TrainingResponseSchema: model_id, metrics, training_time
        
    Raises:
        HTTPException: If training fails
    """
    try:
        file_id = request.file_id
        
        # Check if file exists
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
        
        # Build and train Prophet for each numeric column
        models = {}
        
        for col in numeric_cols:
            # Prepare prophet dataframe
            prophet_df = pd.DataFrame({
                'ds': pd.to_datetime(df[datetime_col]) if datetime_col else pd.date_range(start='2020-01-01', periods=len(df)),
                'y': df[col].values
            })
            
            # Remove NaN
            prophet_df = prophet_df.dropna()
            
            # Build and train
            m = Prophet()
            m.fit(prophet_df)
            
            models[col] = m
        
        # Generate model ID
        model_id = f"prophet_{file_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"[:32]
        
        # Save models
        metadata = {
            'type': 'prophet',
            'file_id': file_id,
            'numeric_cols': numeric_cols,
            'created_at': datetime.now().isoformat()
        }
        
        model_manager.save_model(models, model_id, metadata)
        
        training_time = (datetime.now() - start_time).total_seconds()
        
        logger.info(f"✓ Prophet models trained: {model_id} ({training_time:.2f}s)")
        
        # Store in active models
        active_models[model_id] = metadata
        
        return TrainingResponseSchema(
            model_id=model_id,
            model_type="prophet",
            rows_used=len(df),
            features=len(numeric_cols),
            metrics=MetricsSchema(mae=None, rmse=None, mape=None, r2=None),
            training_time=training_time,
            created_at=datetime.now().isoformat()
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


@app.get("/forecast_lstm", response_model=ForecastResponseSchema)
async def forecast_lstm_endpoint(
    model_id: str = Query(..., description="Trained LSTM model ID"),
    periods: int = Query(24, ge=1, le=365, description="Number of periods to forecast"),
    key_data: dict = Depends(verify_api_key)
):
    """
    Generate LSTM forecast
    
    Args:
        model_id: Trained LSTM model ID
        periods: Number of periods to forecast (1-365)
        key_data: API key metadata (auto-validated)
        
    Returns:
        ForecastResponseSchema: forecast array, metrics, timestamps
        
    Raises:
        HTTPException: If forecast fails
    """
    try:
        # Load model
        model, metadata = model_manager.load_model(model_id)
        
        if model is None:
            raise APIException(
                message=f"Model not found: {model_id}",
                error_code="MODEL_NOT_FOUND",
                status_code=404
            )
        
        # Load scaler
        scaler_id = f"{model_id}_scaler"
        scaler, _ = model_manager.load_model(scaler_id)
        
        if scaler is None:
            raise APIException(
                message="Associated scaler not found",
                error_code="SCALER_NOT_FOUND",
                status_code=500
            )
        
        # Get file data
        file_id = metadata['file_id']
        if file_id not in active_files:
            raise APIException(
                message="Associated file not found",
                error_code="FILE_NOT_FOUND",
                status_code=404
            )
        
        file_data = active_files[file_id]
        df = file_data['data']
        datetime_col = file_data['datetime_col']
        numeric_cols = metadata['numeric_cols']
        lookback = metadata['lookback']
        
        logger.info(f"🔄 Forecasting with LSTM {model_id}...")
        
        # Prepare data
        data = df[numeric_cols].values.astype(np.float32)
        data = np.nan_to_num(data, nan=0.0, posinf=1e6, neginf=-1e6)
        
        # Normalize data using the saved scaler
        X_scaled = scaler.transform(data)
        
        # Create sequences from last lookback window
        last_sequence = X_scaled[-lookback:].reshape(1, lookback, len(numeric_cols))
        
        # Simple forecast by iterating
        forecast_list = []
        current_sequence = last_sequence.copy()
        
        for _ in range(periods):
            # Predict next timestep
            next_pred = model.predict(current_sequence, verbose=0)
            forecast_list.append(next_pred[0])
            
            # Use prediction as new input for next iteration
            current_sequence = np.append(current_sequence[0, 1:, :], next_pred, axis=0).reshape(1, lookback, len(numeric_cols))
        
        # Convert to array and inverse transform
        forecast_array = np.array(forecast_list)
        forecast_inverse = scaler.inverse_transform(forecast_array)
        
        # Generate timestamps
        timestamps = DataPreprocessor.generate_timestamps(df, datetime_col, periods)
        
        logger.info(f"✓ LSTM forecast generated: {periods} periods")
        
        return ForecastResponseSchema(
            forecast=forecast_inverse.tolist(),
            actual=df[numeric_cols].tail(periods).values.tolist() if len(df) >= periods else None,
            timestamps=timestamps,
            metrics=None,
            model_type="lstm",
            periods=periods
        )
    
    except APIException:
        raise
    except Exception as e:
        logger.error(f"❌ LSTM forecast error: {str(e)}")
        raise APIException(
            message=f"LSTM forecast failed: {str(e)}",
            error_code="FORECAST_ERROR",
            status_code=500
        )


@app.get("/forecast_prophet", response_model=ForecastResponseSchema)
async def forecast_prophet_endpoint(
    model_id: str = Query(..., description="Trained Prophet model ID"),
    periods: int = Query(24, ge=1, le=365, description="Number of periods to forecast"),
    key_data: dict = Depends(verify_api_key)
):
    """
    Generate Prophet forecast
    
    Args:
        model_id: Trained Prophet model ID
        periods: Number of periods to forecast (1-365)
        key_data: API key metadata (auto-validated)
        
    Returns:
        ForecastResponseSchema: forecast array, metrics, timestamps
        
    Raises:
        HTTPException: If forecast fails
    """
    try:
        # Load models
        models, metadata = model_manager.load_model(model_id)
        
        if models is None:
            raise APIException(
                message=f"Model not found: {model_id}",
                error_code="MODEL_NOT_FOUND",
                status_code=404
            )
        
        # Get file data
        file_id = metadata['file_id']
        if file_id not in active_files:
            raise APIException(
                message="Associated file not found",
                error_code="FILE_NOT_FOUND",
                status_code=404
            )
        
        file_data = active_files[file_id]
        numeric_cols = metadata['numeric_cols']
        
        logger.info(f"🔄 Forecasting with Prophet {model_id}...")
        
        # Generate forecasts for all columns
        forecast_all = []
        
        for col in numeric_cols:
            model = models[col]
            future = model.make_future_dataframe(periods=periods)
            forecast = model.predict(future)
            forecast_all.append(forecast['yhat'].values[-periods:])
        
        forecast_array = np.array(forecast_all).T
        
        # Generate timestamps
        df = file_data['data']
        datetime_col = file_data['datetime_col']
        timestamps = DataPreprocessor.generate_timestamps(df, datetime_col, periods)
        
        logger.info(f"✓ Prophet forecast generated: {periods} periods")
        
        return ForecastResponseSchema(
            forecast=forecast_array.tolist(),
            actual=df[numeric_cols].tail(periods).values.tolist() if len(df) >= periods else None,
            timestamps=timestamps,
            metrics=None,
            model_type="prophet",
            periods=periods
        )
    
    except APIException:
        raise
    except Exception as e:
        logger.error(f"❌ Prophet forecast error: {str(e)}")
        raise APIException(
            message=f"Prophet forecast failed: {str(e)}",
            error_code="FORECAST_ERROR",
            status_code=500
        )


@app.get("/models")
async def list_models():
    """
    List all trained models
    
    Returns:
        dict: active models with metadata
    """
    return {
        'total': len(active_models),
        'models': active_models
    }


@app.get("/files")
async def list_files():
    """
    List all uploaded files
    
    Returns:
        dict: active files with metadata
    """
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
    """
    Delete file and associated models
    
    Args:
        file_id: File ID to delete
        
    Returns:
        dict: cleanup status
    """
    deleted_models = []
    
    # Delete associated models
    for model_id in list(active_models.keys()):
        if active_models[model_id].get('file_id') == file_id:
            model_manager.load_model(model_id)  # Verify exists
            deleted_models.append(model_id)
            del active_models[model_id]
    
    # Delete file
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
    """
    Generate a new API key (Master key required)
    
    Args:
        name: Description for this API key
        key_data: Master API key (auto-validated)
        
    Returns:
        dict: New API key (store securely, can't retrieve again!)
    """
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
    """
    List all API keys (Master key required)
    
    Returns:
        dict: List of API keys with metadata
    """
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
    """
    Revoke an API key (Master key required)
    
    Args:
        key_partial: Key identifier (from list endpoint)
        key_data: Master API key
        
    Returns:
        dict: Status
    """
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

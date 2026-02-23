from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import numpy as np
import pandas as pd
import io
import os
from datetime import datetime, timedelta
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.data_preprocessing import DataPreprocessor
from src.models.lstm_model import train_lstm, LSTMModel
from src.evaluation import calculate_metrics

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


class PredictionResponse(BaseModel):
    success: bool
    message: str
    forecast: List[float]
    actual: Optional[List[float]] = None
    timestamps: List[str]
    metrics: Dict[str, Any]
    model_type: str


class TrainingResponse(BaseModel):
    success: bool
    message: str
    model_name: str
    metrics: Dict[str, float]
    data_shapes: Dict[str, Any]


class DataManager:
    def __init__(self):
        self.data = {}
        self.models = {}
    
    def load_csv(self, file: UploadFile) -> pd.DataFrame:
        contents = file.file.read()
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        self.data[file.filename] = df
        return df
    
    def get_data(self, filename: str) -> pd.DataFrame:
        if filename not in self.data:
            raise ValueError(f"Arquivo '{filename}' não encontrado")
        return self.data[filename]
    
    def save_model(self, name: str, model: Any):
        self.models[name] = model


app = FastAPI(
    title="TOTEM_DEEPSEA API",
    description="Sistema de previsão de séries temporais",
    version="2.0.0"
)

manager = DataManager()
preprocessor = DataPreprocessor()


@app.get("/", tags=["Health"])
async def root():
    return {
        "status": "✅ online",
        "version": "2.0.0",
        "models": len(manager.models),
        "datasets": len(manager.data)
    }


@app.post("/upload", tags=["Data"])
async def upload_file(file: UploadFile = File(...)):
    try:
        if not file.filename.endswith('.csv'):
            raise ValueError("Apenas arquivos CSV")
        
        df = manager.load_csv(file)
        
        return JSONResponse({
            "success": True,
            "filename": file.filename,
            "rows": df.shape[0],
            "columns": df.shape[1],
            "column_names": list(df.columns)
        })
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/train", tags=["Training"])
async def train_model(
    filename: str,
    epochs: int = Query(30, ge=1, le=200),
    lookback: int = Query(30, ge=5, le=120),
    batch_size: int = Query(32, ge=1, le=128)
):
    try:
        df = manager.get_data(filename)
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) == 0:
            raise ValueError("Nenhuma coluna numérica")
        
        data = df[numeric_cols].values
        X_norm, _, _ = preprocessor.normalize_data(data, None, None)
        X_seq, y_seq = preprocessor.create_sequences(X_norm, lookback=lookback)
        
        X_train, X_val, X_test, y_train, y_val, y_test = \
            preprocessor.train_test_split(X_seq, y_seq)
        
        y_train_single = y_train[:, 0]
        y_val_single = y_val[:, 0]
        y_test_single = y_test[:, 0]
        
        model, history = train_lstm(
            X_train, y_train_single,
            X_val, y_val_single,
            epochs=epochs,
            batch_size=batch_size
        )
        
        y_pred = model.predict(X_test, verbose=0).flatten()
        metrics = calculate_metrics(y_test_single, y_pred)
        
        model_name = f"{filename.replace('.csv', '')}_lstm"
        manager.save_model(model_name, model)
        
        return TrainingResponse(
            success=True,
            message="✅ Modelo treinado",
            model_name=model_name,
            metrics={
                "mae": float(metrics.get('mae', 0)),
                "rmse": float(metrics.get('rmse', 0)),
                "mape": float(metrics.get('mape', 0)),
                "r2": float(metrics.get('r2', 0))
            },
            data_shapes={
                "train": list(X_train.shape),
                "val": list(X_val.shape),
                "test": list(X_test.shape)
            }
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/predict", tags=["Prediction"])
async def predict(
    filename: str,
    model_name: str,
    periods: int = Query(24, ge=1, le=365)
):
    try:
        if model_name not in manager.models:
            raise ValueError(f"Modelo '{model_name}' não encontrado")
        
        model = manager.models[model_name]
        df = manager.get_data(filename)
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        data = df[numeric_cols].values
        
        X_norm, _, _ = preprocessor.normalize_data(data, None, None)
        X_seq, y_seq = preprocessor.create_sequences(X_norm, lookback=30)
        
        X_base = X_norm[-30:].reshape(1, 30, -1)
        
        predictions = []
        current = X_base.copy()
        
        for _ in range(periods):
            pred = model.predict(current, verbose=0)[0]
            predictions.append(float(pred[0]))
            current = np.concatenate([current[0, 1:, :], pred.reshape(1, -1)], axis=0).reshape(1, 30, -1)
        
        now = datetime.now()
        timestamps = [(now + timedelta(hours=i)).isoformat() for i in range(1, periods + 1)]
        
        return PredictionResponse(
            success=True,
            message="✅ Previsão gerada",
            forecast=predictions,
            actual=data[-periods:, 0].tolist() if len(data) >= periods else None,
            timestamps=timestamps,
            metrics={
                "forecast_periods": periods,
                "model_type": "LSTM"
            },
            model_type="LSTM"
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/models", tags=["Info"])
async def list_models():
    return {
        "success": True,
        "total": len(manager.models),
        "models": list(manager.models.keys())
    }


@app.get("/health", tags=["Health"])
async def health():
    return {
        "status": "✅ Healthy",
        "timestamp": datetime.now().isoformat()
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

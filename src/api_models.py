"""
API Models - Pydantic schemas for request/response validation
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import json


class UploadResponseSchema(BaseModel):
    """Response for CSV upload endpoint"""
    file_id: str
    rows: int
    columns: List[str]
    datetime_column: Optional[str] = None
    numeric_columns: List[str]
    
    class Config:
        json_schema_extra = {
            "example": {
                "message": "✓ File uploaded and validated successfully",
                "file_id": "abc123def456",
                "rows": 500,
                "columns": ["timestamp", "temperature", "humidity", "pressure"],
                "datetime_column": "timestamp",
                "numeric_columns": ["temperature", "humidity", "pressure"]
            }
        }


class TrainingRequestSchema(BaseModel):
    """Request for model training"""
    file_id: str
    lookback: int = Field(default=24, ge=5, le=500)
    test_size: float = Field(default=0.2, ge=0.1, le=0.5)
    epochs: int = Field(default=50, ge=5, le=500)
    batch_size: int = Field(default=16, ge=4, le=128)
    
    class Config:
        json_schema_extra = {
            "example": {
                "file_id": "abc123def456",
                "lookback": 24,
                "test_size": 0.2,
                "epochs": 50,
                "batch_size": 16
            }
        }


class TrainingResponseSchema(BaseModel):
    """Response for model training"""
    model_id: str
    model_type: str
    training_time: float
    metrics: Optional["MetricsSchema"] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "message": "✓ LSTM model trained successfully",
                "model_type": "LSTM",
                "model_id": "lstm_abc123",
                "training_time": 45.23,
                "metrics": {
                    "val_loss": 0.0234,
                    "val_mae": 0.1523
                },
                "params": {
                    "lookback": 24,
                    "epochs": 50,
                    "batch_size": 16
                }
            }
        }


class ForecastRequestSchema(BaseModel):
    """Request for forecast"""
    model_id: str
    periods: int = Field(default=24, ge=1, le=500)
    
    class Config:
        json_schema_extra = {
            "example": {
                "model_id": "lstm_abc123",
                "periods": 24
            }
        }


class MetricsSchema(BaseModel):
    """Model evaluation metrics"""
    mae: Optional[float] = Field(None, description="Mean Absolute Error")
    rmse: Optional[float] = Field(None, description="Root Mean Squared Error")
    mape: Optional[float] = Field(None, description="Mean Absolute Percentage Error")
    r2: Optional[float] = Field(None, description="R² Score")
    
    class Config:
        json_schema_extra = {
            "example": {
                "mae": 0.1523,
                "rmse": 0.1847,
                "mape": 2.34,
                "r2": 0.85
            }
        }


class ForecastResponseSchema(BaseModel):
    """Response for forecast endpoint"""
    forecast: List[List[float]] = Field(..., description="Predicted values")
    actual: Optional[List[List[float]]] = Field(None, description="Actual values if available")
    timestamps: List[str] = Field(..., description="ISO format timestamps")
    metrics: Optional[MetricsSchema] = None
    model_type: str
    periods: int
    message: str
    
    class Config:
        json_schema_extra = {
            "example": {
                "forecast": [[20.5, 65.2, 1013.2], [20.8, 65.0, 1013.1]],
                "actual": None,
                "timestamps": ["2024-01-01T00:00:00", "2024-01-01T01:00:00"],
                "metrics": {
                    "mae": 0.1523,
                    "rmse": 0.1847,
                    "mape": 2.34,
                    "r2": 0.85
                },
                "model_type": "LSTM",
                "periods": 24,
                "message": "✓ Forecast generated successfully"
            }
        }


class ErrorResponseSchema(BaseModel):
    """Error response schema"""
    error: str
    message: Optional[str] = None
    error_code: str
    timestamp: str
    
    class Config:
        json_schema_extra = {
            "example": {
                "error": "File not found",
                "detail": "file_id 'abc123' does not exist",
                "error_code": "FILE_NOT_FOUND"
            }
        }


class HealthCheckSchema(BaseModel):
    """Health check response"""
    status: str
    version: str
    timestamp: str
    
    class Config:
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "version": "1.0.0",
                "timestamp": "2024-01-01T12:00:00"
            }
        }

"""
QUICK START EXAMPLE - TOTEM_DEEPSEA
This script demonstrates how to use TOTEM_DEEPSEA for time series forecasting
"""

import sys
import os
import numpy as np
import pandas as pd

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.config import RAW_PATH
from src.data_preprocessing import quick_preprocess, DataPreprocessor
from src.models.lstm_model import train_lstm
from src.evaluation import calculate_metrics, plot_predictions, create_evaluation_report
from src.prediction import PredictionEngine
from dashboard.plotly_charts import create_prediction_vs_actual


def create_sample_data():
    """
    Create sample data for demonstration
    This would normally be your real CSV file
    """
    print("=" * 60)
    print("📊 CREATING SAMPLE DATA")
    print("=" * 60)
    
    # Generate synthetic time series data
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=500, freq='h')
    
    # Create multivariate time series
    data = {
        'timestamp': dates,
        'temperature': 20 + 5 * np.sin(np.arange(500) * 2 * np.pi / 24) + np.random.randn(500) * 0.5,
        'humidity': 60 + 15 * np.sin(np.arange(500) * 2 * np.pi / 24 + 1) + np.random.randn(500) * 1,
        'pressure': 1013 + 2 * np.sin(np.arange(500) * 2 * np.pi / 168) + np.random.randn(500) * 0.3,
    }
    
    df = pd.DataFrame(data)
    
    # Save to CSV using RAW_PATH from config
    csv_path = os.path.join(RAW_PATH, 'sample_data.csv')
    df.to_csv(csv_path, index=False)
    
    print(f"✓ Sample data created: {csv_path}")
    print(f"  Shape: {df.shape}")
    print(f"\nFirst few rows:")
    print(df.head())
    print()
    
    return csv_path


def run_full_pipeline():
    """Run the complete forecasting pipeline"""
    
    print("\n" + "=" * 60)
    print("🚀 TOTEM_DEEPSEA - FULL PIPELINE")
    print("=" * 60 + "\n")
    
    # Step 1: Create sample data
    print("STEP 1: Data Preparation")
    print("-" * 60)
    csv_path = create_sample_data()
    
    # Step 2: Preprocess data
    print("\nSTEP 2: Data Preprocessing")
    print("-" * 60)
    
    try:
        X_train, X_val, X_test, y_train, y_val, y_test, scaler = \
            quick_preprocess('sample_data.csv', lookback=24, datetime_col='timestamp')
        
        print(f"✓ Data preprocessed successfully")
        print(f"  X_train shape: {X_train.shape}")
        print(f"  X_test shape: {X_test.shape}")
        print(f"  y_train shape: {y_train.shape}")
    
    except Exception as e:
        print(f"❌ Preprocessing error: {e}")
        return
    
    # Step 3: Train LSTM model
    print("\nSTEP 3: Training LSTM Model")
    print("-" * 60)
    
    try:
        print("Training... (this may take a minute)")
        model, history = train_lstm(
            X_train, y_train,
            X_val, y_val,
            model_name='example_model'
        )
        
        print("✓ LSTM model trained successfully")
        print(f"  Final training loss: {history.history['loss'][-1]:.6f}")
        print(f"  Final validation loss: {history.history['val_loss'][-1]:.6f}")
    
    except Exception as e:
        print(f"❌ Training error: {e}")
        return
    
    # Step 4: Make predictions
    print("\nSTEP 4: Making Predictions")
    print("-" * 60)
    
    try:
        y_pred_train = model.predict(X_train, verbose=0)
        y_pred_val = model.predict(X_val, verbose=0)
        y_pred_test = model.predict(X_test, verbose=0)
        
        # Inverse scale predictions
        y_pred_train = scaler.inverse_transform(y_pred_train)
        y_pred_val = scaler.inverse_transform(y_pred_val)
        y_pred_test = scaler.inverse_transform(y_pred_test)
        
        y_train_orig = scaler.inverse_transform(y_train)
        y_val_orig = scaler.inverse_transform(y_val)
        y_test_orig = scaler.inverse_transform(y_test)
        
        print("✓ Predictions generated")
        print(f"  Train predictions shape: {y_pred_train.shape}")
        print(f"  Test predictions shape: {y_pred_test.shape}")
    
    except Exception as e:
        print(f"❌ Prediction error: {e}")
        return
    
    # Step 5: Evaluate model
    print("\nSTEP 5: Model Evaluation")
    print("-" * 60)
    
    try:
        # Calculate metrics for test set
        metrics = calculate_metrics(y_test_orig, y_pred_test)
        
        print("✓ Evaluation metrics calculated:")
        for metric_name, metric_value in metrics.items():
            print(f"  {metric_name.upper()}: {metric_value:.6f}")
        
        # Create evaluation report
        report = create_evaluation_report(
            y_test_orig, y_pred_test,
            model_name='example_model'
        )
    
    except Exception as e:
        print(f"❌ Evaluation error: {e}")
        return
    
    # Step 6: Use Prediction Engine
    print("\nSTEP 6: Using Prediction Engine")
    print("-" * 60)
    
    try:
        engine = PredictionEngine('example_model')
        
        # Make new predictions
        new_predictions = engine.predict_lstm(X_test[:10], inverse_scale=True)
        
        print("✓ PredictionEngine predictions generated")
        print(f"  Input shape: {X_test[:10].shape}")
        print(f"  Output shape: {new_predictions.shape}")
        print(f"\nSample predictions (first 3):")
        print(new_predictions[:3])
    
    except Exception as e:
        print(f"⚠️  PredictionEngine warning: {e}")
    
    # Summary
    print("\n" + "=" * 60)
    print("✅ PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Launch the dashboard: streamlit run dashboard/streamlit_app.py")
    print("2. Upload your own CSV data to data/raw/")
    print("3. Train models with your data")
    print("4. Make predictions and visualize results")
    print()


def quick_example_lstm():
    """Quick example showing LSTM model usage"""
    
    print("\n" + "=" * 60)
    print("⚡ QUICK LSTM EXAMPLE")
    print("=" * 60 + "\n")
    
    from models.lstm_model import build_lstm_model, train_lstm
    
    # Create dummy data
    np.random.seed(42)
    X_train = np.random.randn(100, 24, 3)
    y_train = np.random.randn(100, 3)
    X_val = np.random.randn(20, 24, 3)
    y_val = np.random.randn(20, 3)
    
    print("Sample data created:")
    print(f"  X_train: {X_train.shape} (100 samples, 24 timesteps, 3 features)")
    print(f"  y_train: {y_train.shape}")
    
    # Build model
    print("\nBuilding LSTM model...")
    model = build_lstm_model((24, 3))
    print(f"✓ Model built with {model.count_params():,} parameters")
    
    # Make predictions without training
    print("\nMaking predictions...")
    predictions = model.predict(X_train[:5], verbose=0)
    print(f"✓ Predictions shape: {predictions.shape}")
    
    print("\n✅ Quick example completed!")
    print()


def quick_example_prophet():
    """Quick example showing Prophet model usage"""
    
    print("\n" + "=" * 60)
    print("🔮 QUICK PROPHET EXAMPLE")
    print("=" * 60 + "\n")
    
    try:
        from models.prophet_model import train_prophet, forecast_prophet
        
        # Create sample time series
        dates = pd.date_range('2024-01-01', periods=100)
        df = pd.DataFrame({
            'timestamp': dates,
            'values': 20 + 5 * np.sin(np.arange(100) * 2 * np.pi / 24) + np.random.randn(100) * 0.5
        })
        df.set_index('timestamp', inplace=True)
        
        print("Sample time series created:")
        print(f"  Shape: {df.shape}")
        print(f"  Date range: {df.index[0]} to {df.index[-1]}")
        
        # Train Prophet model
        print("\nTraining Prophet model...")
        model = train_prophet(df, col_target='values', col_datetime='timestamp')
        print("✓ Prophet model trained")
        
        # Make forecast
        print("\nGenerating forecast for next 24 periods...")
        forecast = forecast_prophet(model, periods=24)
        print("✓ Forecast generated")
        print(f"\nForecast preview:")
        print(forecast.head(5))
        
        print("\n✅ Prophet example completed!")
    
    except ImportError:
        print("⚠️  Prophet is not installed. Install with: pip install prophet")
    
    print()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='TOTEM_DEEPSEA Quick Start')
    parser.add_argument('--full', action='store_true', help='Run full pipeline')
    parser.add_argument('--lstm', action='store_true', help='Quick LSTM example')
    parser.add_argument('--prophet', action='store_true', help='Quick Prophet example')
    
    args = parser.parse_args()
    
    # Run all by default if no args specified
    if not any([args.full, args.lstm, args.prophet]):
        run_full_pipeline()
    else:
        if args.full:
            run_full_pipeline()
        if args.lstm:
            quick_example_lstm()
        if args.prophet:
            quick_example_prophet()
    
    print("\n📚 For more information, see README.md")

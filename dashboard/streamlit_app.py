"""
TOTEM_DEEPSEA - Streamlit Dashboard
Interactive dashboard for time series forecasting
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import sys

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.data_preprocessing import DataPreprocessor
from src.prediction import PredictionEngine
from src.evaluation import calculate_metrics, plot_predictions, create_evaluation_report
from dashboard.plotly_charts import (
    create_time_series_plot, create_forecast_plot,
    create_prediction_vs_actual, create_metrics_comparison
)
from src.config import RAW_PATH, MODELS_PATH, FORECAST_HORIZON

# Configure page
st.set_page_config(
    page_title='TOTEM_DEEPSEA',
    page_icon='🔮',
    layout='wide',
    initial_sidebar_state='expanded'
)

# Custom styling
st.markdown("""
    <style>
        .main-header {
            font-size: 3em;
            color: #00D9FF;
            text-align: center;
            font-weight: bold;
            text-shadow: 2px 2px 4px rgba(0,217,255,0.5);
        }
        .section-header {
            font-size: 2em;
            color: #00FF84;
            margin-top: 2em;
            border-bottom: 2px solid #00FF84;
            padding-bottom: 0.5em;
        }
        .metric-card {
            background: linear-gradient(135deg, #1E3C72 0%, #2A5298 100%);
            padding: 20px;
            border-radius: 10px;
            border-left: 5px solid #00D9FF;
        }
    </style>
""", unsafe_allow_html=True)


def main():
    """Main Streamlit app"""
    
    # Header
    st.markdown('<div class="main-header">🔮 TOTEM_DEEPSEA</div>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; color: #00D9FF; font-size: 1.1em;">Multivariate Time Series Forecasting System</p>', unsafe_allow_html=True)
    
    st.divider()
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        menu = st.radio(
            "Select Module:",
            ["📊 Data Exploration", "🤖 Model Training", "🔮 Predictions", "📈 Evaluation"]
        )
    
    # ============================================================
    # DATA EXPLORATION
    # ============================================================
    if menu == "📊 Data Exploration":
        st.markdown('<h2 class="section-header">Data Exploration</h2>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # File selection
            csv_files = [f for f in os.listdir(RAW_PATH) if f.endswith('.csv')]
            
            if csv_files:
                selected_file = st.selectbox("Select CSV file:", csv_files)
                
                # Load data
                preprocessor = DataPreprocessor()
                df = preprocessor.load_raw_data(selected_file)
                
                st.success(f"✓ Loaded {len(df)} rows, {len(df.columns)} columns")
            else:
                st.warning("No CSV files found in data/raw/")
                st.info("Upload CSV files to the data/raw/ directory to get started")
                return
        
        with col2:
            st.metric("Total Rows", len(df))
            st.metric("Total Columns", len(df.columns))
        
        # Data preview
        st.subheader("Data Preview")
        st.dataframe(df.head(20), use_container_width=True)
        
        # Data statistics
        st.subheader("Statistical Summary")
        st.dataframe(df.describe(), use_container_width=True)
        
        # Column selection for visualization
        st.subheader("Time Series Visualization")
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        selected_cols = st.multiselect(
            "Select columns to visualize:",
            numeric_cols,
            default=numeric_cols[:min(3, len(numeric_cols))]
        )
        
        if selected_cols:
            # Create and display plot
            fig = create_time_series_plot(df, selected_cols, title="Time Series Data")
            st.plotly_chart(fig, use_container_width=True)
    
    # ============================================================
    # MODEL TRAINING
    # ============================================================
    elif menu == "🤖 Model Training":
        st.markdown('<h2 class="section-header">Model Training</h2>', unsafe_allow_html=True)
        
        st.warning("⚠️ Model training is typically done via terminal or notebooks.")
        st.info("""
        To train models:
        1. Prepare your data in `data/raw/` folder
        2. Run: `python -m src.models.train`
        3. Or use Jupyter notebook: `notebooks/train_model.ipynb`
        
        Training will:
        - Preprocess your data
        - Train LSTM neural network
        - Train Prophet models
        - Save models in `src/models/saved/`
        """)
        
        # Show available trained models
        st.subheader("Available Trained Models")
        model_files = [f for f in os.listdir(MODELS_PATH) if f.endswith('.h5')]
        
        if model_files:
            st.success(f"✓ Found {len(model_files)} trained models")
            for model_file in model_files:
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.write(f"📦 {model_file}")
                with col2:
                    st.write(f"✅ Ready")
        else:
            st.warning("No trained models found. Train a model first!")
    
    # ============================================================
    # PREDICTIONS
    # ============================================================
    elif menu == "🔮 Predictions":
        st.markdown('<h2 class="section-header">Make Predictions</h2>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Model selection
            model_files = [f.replace('_lstm_model.h5', '').replace('_best.h5', '') 
                          for f in os.listdir(MODELS_PATH) if f.endswith('_best.h5')]
            
            if not model_files:
                st.warning("No trained models available. Train a model first!")
                return
            
            selected_model = st.selectbox("Select trained model:", model_files)
        
        with col2:
            forecast_periods = st.number_input("Forecast periods:", 1, 168, FORECAST_HORIZON)
        
        st.divider()
        
        # Load prediction engine
        try:
            engine = PredictionEngine(selected_model)
            st.success("✓ Prediction engine loaded")
        except Exception as e:
            st.error(f"❌ Error loading model: {e}")
            return
        
        # CSV file selection for prediction
        csv_files = [f for f in os.listdir(RAW_PATH) if f.endswith('.csv')]
        selected_csv = st.selectbox("Select data for prediction:", csv_files) if csv_files else None
        
        if selected_csv and st.button("🚀 Generate Predictions"):
            with st.spinner("Generating predictions..."):
                try:
                    predictions = engine.predict_from_csv(selected_csv)
                    st.success("✅ Predictions generated!")
                    
                    # Display LSTM predictions
                    if 'lstm' in predictions:
                        st.subheader("LSTM Predictions")
                        lstm_pred = predictions['lstm']['predictions']
                        st.write(f"Shape: {lstm_pred.shape}")
                        st.text(f"First 5 predictions:\n{lstm_pred[:5]}")
                    
                    # Display Prophet predictions
                    if 'prophet' in predictions:
                        st.subheader("Prophet Predictions")
                        for col_name, forecast_df in predictions['prophet'].items():
                            st.write(f"**{col_name}**")
                            st.dataframe(forecast_df, use_container_width=True)
                
                except Exception as e:
                    st.error(f"❌ Prediction error: {e}")
    
    # ============================================================
    # EVALUATION
    # ============================================================
    elif menu == "📈 Evaluation":
        st.markdown('<h2 class="section-header">Model Evaluation</h2>', unsafe_allow_html=True)
        
        # Load test data
        st.subheader("Load Test Data")
        csv_files = [f for f in os.listdir(RAW_PATH) if f.endswith('.csv')]
        
        if csv_files:
            selected_file = st.selectbox("Select test data:", csv_files)
            
            preprocessor = DataPreprocessor()
            df = preprocessor.load_raw_data(selected_file)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Records", len(df))
            with col2:
                st.metric("Columns", len(df.columns))
            with col3:
                st.metric("Memory Usage", f"{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
            
            st.divider()
            
            # Model evaluation
            st.subheader("Evaluate Trained Model")
            
            model_files = [f.replace('_best.h5', '') 
                          for f in os.listdir(MODELS_PATH) if f.endswith('_best.h5')]
            
            if model_files:
                selected_model = st.selectbox("Select model to evaluate:", model_files)
                
                if st.button("📊 Run Evaluation"):
                    with st.spinner("Running evaluation..."):
                        try:
                            engine = PredictionEngine(selected_model)
                            st.success("✓ Model loaded for evaluation")
                            
                            st.info("📝 Full evaluation report would include:")
                            st.write("- MAE, RMSE, MAPE, R² metrics")
                            st.write("- Prediction vs Actual plots")
                            st.write("- Error distribution analysis")
                            st.write("- Residuals analysis")
                            
                        except Exception as e:
                            st.error(f"❌ Evaluation error: {e}")
            else:
                st.warning("No trained models available!")
        else:
            st.warning("No CSV files found in data/raw/")
    
    st.divider()
    
    # Footer
    st.markdown("""
    ---
    <div style="text-align: center; color: #888; margin-top: 2em;">
        <p>🔮 TOTEM_DEEPSEA v1.0 | Multivariate Time Series Forecasting</p>
        <p style="font-size: 0.9em;">Built with TensorFlow, Prophet, and Streamlit</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()

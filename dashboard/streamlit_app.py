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
from src.stock_analysis import StockAnalyzer, get_brazilian_stocks_list, get_us_stocks_list
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
            [
                "📊 Data Exploration", 
                "🤖 Model Training", 
                "🔮 Predictions", 
                "📈 Evaluation",
                "📈 Stock Analysis",
                "💡 Stock Recommendations"
            ]
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
        st.dataframe(df.head(20), width='stretch')
        
        # Data statistics
        st.subheader("Statistical Summary")
        st.dataframe(df.describe(), width='stretch')
        
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
                            st.dataframe(forecast_df, width='stretch')
                
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
    
    # ============================================================
    # STOCK ANALYSIS
    # ============================================================
    elif menu == "📈 Stock Analysis":
        st.markdown('<h2 class="section-header">📈 Stock Analysis & Price Prediction</h2>', unsafe_allow_html=True)
        
        analyzer = StockAnalyzer()
        
        col1, col2 = st.columns(2)
        
        with col1:
            market = st.segmented_control(
                "Market:",
                ["🇺🇸 US Market", "🇧🇷 Brazil Market"],
                default="🇺🇸 US Market"
            )
            
            if market == "🇺🇸 US Market":
                available_stocks = get_us_stocks_list()
            else:
                available_stocks = get_brazilian_stocks_list()
        
        with col2:
            selected_stock = st.selectbox("Select Stock Ticker:", available_stocks)
        
        st.divider()
        
        if st.button("📥 Fetch Stock Data", use_container_width=True):
            with st.spinner(f"Fetching data for {selected_stock}..."):
                try:
                    # Fetch data
                    df = analyzer.fetch_stock_data(selected_stock, period='6mo')
                    
                    if df is not None:
                        # Add indicators
                        df = analyzer.add_technical_indicators(df)
                        
                        # Calculate metrics
                        metrics = analyzer.calculate_metrics(df)
                        
                        # Display metrics
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric(
                                "Current Price",
                                f"${metrics['Current Price']:.2f}",
                                f"{metrics['Day Change %']:.2f}%"
                            )
                        
                        with col2:
                            st.metric("52W High", f"${metrics['52 Week High']:.2f}")
                        
                        with col3:
                            st.metric("52W Low", f"${metrics['52 Week Low']:.2f}")
                        
                        with col4:
                            signal = analyzer.calculate_signal(df)
                            st.metric("Signal", signal, "⭐")
                        
                        st.divider()
                        
                        # Chart
                        st.subheader("Price Chart with Technical Indicators")
                        
                        import plotly.graph_objects as go
                        
                        fig = go.Figure()
                        
                        # Price
                        fig.add_trace(go.Scatter(
                            x=df.index,
                            y=df['Close'],
                            name='Close',
                            line=dict(color='#00D9FF', width=2)
                        ))
                        
                        # SMA 20
                        fig.add_trace(go.Scatter(
                            x=df.index,
                            y=df['SMA_20'],
                            name='SMA 20',
                            line=dict(color='orange', width=1)
                        ))
                        
                        # SMA 50
                        fig.add_trace(go.Scatter(
                            x=df.index,
                            y=df['SMA_50'],
                            name='SMA 50',
                            line=dict(color='red', width=1)
                        ))
                        
                        # Bollinger Bands
                        fig.add_trace(go.Scatter(
                            x=df.index,
                            y=df['BB_Upper'],
                            name='BB Upper',
                            line=dict(color='rgba(100,100,100,0.3)', width=1),
                            showlegend=False
                        ))
                        
                        fig.add_trace(go.Scatter(
                            x=df.index,
                            y=df['BB_Lower'],
                            name='BB Lower',
                            line=dict(color='rgba(100,100,100,0.3)', width=1),
                            fill='tonexty',
                            fillcolor='rgba(100,100,100,0.1)'
                        ))
                        
                        fig.update_layout(
                            title=f"{selected_stock} - Technical Analysis",
                            xaxis_title="Date",
                            yaxis_title="Price (USD)" if market == "🇺🇸 US Market" else "Price (BRL)",
                            template='plotly_dark',
                            height=500,
                            hovermode='x unified'
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        st.divider()
                        
                        # Indicators detail
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("RSI", f"{df['RSI'].iloc[-1]:.1f}", 
                                     "Oversold" if df['RSI'].iloc[-1] < 30 else 
                                     "Overbought" if df['RSI'].iloc[-1] > 70 else 
                                     "Neutral")
                        
                        with col2:
                            st.metric("MACD", f"{df['MACD'].iloc[-1]:.4f}",
                                     "Bullish ↑" if df['MACD'].iloc[-1] > df['MACD_Signal'].iloc[-1] 
                                     else "Bearish ↓")
                        
                        with col3:
                            avg_vol = df['Volume'].tail(30).mean()
                            current_vol = df['Volume'].iloc[-1]
                            st.metric("Volume", f"{int(current_vol/1e6)}M",
                                     f"{((current_vol/avg_vol - 1) * 100):.1f}% avg")
                        
                        # Store data in session for prediction
                        st.session_state.stock_data = df
                        st.session_state.selected_stock = selected_stock
                        
                        st.success("✅ Data loaded successfully! Go to Stock Predictions to forecast prices.")
                
                except Exception as e:
                    st.error(f"❌ Error: {e}")
    
    # ============================================================
    # STOCK RECOMMENDATIONS
    # ============================================================
    elif menu == "💡 Stock Recommendations":
        st.markdown('<h2 class="section-header">💡 Buy/Sell Recommendations</h2>', unsafe_allow_html=True)
        
        analyzer = StockAnalyzer()
        
        col1, col2 = st.columns(2)
        
        with col1:
            market = st.segmented_control(
                "Market:",
                ["🇺🇸 US Market", "🇧🇷 Brazil Market"],
                default="🇺🇸 US Market"
            )
        
        with col2:
            analysis_period = st.selectbox(
                "Analysis Period:",
                ["1mo", "3mo", "6mo", "1y"],
                index=2
            )
        
        if st.button("🔍 Analyze Stocks", use_container_width=True):
            with st.spinner("Analyzing all stocks..."):
                try:
                    if market == "🇺🇸 US Market":
                        tickers = get_us_stocks_list()
                    else:
                        tickers = get_brazilian_stocks_list()
                    
                    # Get recommendations
                    recommendations = analyzer.get_stock_recommendations(tickers, period=analysis_period)
                    
                    if not recommendations.empty:
                        st.subheader("📊 Analysis Results")
                        
                        # Color coding based on signal
                        def color_signal(val):
                            if val == 'BUY':
                                return 'background-color: #00FF84'
                            elif val == 'SELL':
                                return 'background-color: #FF6B6B'
                            else:
                                return 'background-color: #FFD700'
                        
                        # Display table with styling
                        styled_df = recommendations.style.applymap(
                            color_signal, subset=['Signal']
                        )
                        
                        st.dataframe(styled_df, width='stretch', height=400)
                        
                        st.divider()
                        
                        # Statistics
                        col1, col2, col3 = st.columns(3)
                        
                        buy_count = len(recommendations[recommendations['Signal'] == 'BUY'])
                        sell_count = len(recommendations[recommendations['Signal'] == 'SELL'])
                        hold_count = len(recommendations[recommendations['Signal'] == 'HOLD'])
                        
                        with col1:
                            st.metric("🟢 BUY Signals", buy_count)
                        
                        with col2:
                            st.metric("🔴 SELL Signals", sell_count)
                        
                        with col3:
                            st.metric("🟡 HOLD Signals", hold_count)
                        
                        st.divider()
                        
                        # Export data
                        st.subheader("📥 Export Results")
                        
                        csv = recommendations.to_csv(index=False)
                        st.download_button(
                            label="Download CSV",
                            data=csv,
                            file_name=f"stock_recommendations_{analysis_period}.csv",
                            mime="text/csv"
                        )
                    
                    else:
                        st.warning("No data available for selected stocks")
                
                except Exception as e:
                    st.error(f"❌ Error: {e}")
        
        st.divider()
        
        # Information
        st.info("""
        ### How Recommendations Work:
        
        Our algorithm analyzes each stock based on:
        
        - **RSI (Relative Strength Index)**: Identifies overbought/oversold conditions
        - **MACD (Moving Average Convergence Divergence)**: Detects trend changes
        - **Bollinger Bands**: Shows price volatility levels
        - **Moving Averages**: Identifies trend direction (20, 50, 200 days)
        - **Volume Analysis**: Confirms price movements
        
        **Signal Meanings:**
        - 🟢 **BUY**: Multiple bullish indicators aligned
        - 🔴 **SELL**: Multiple bearish indicators aligned
        - 🟡 **HOLD**: Mixed signals, wait for clarity
        """)
    
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

"""
Stock Analysis Module
Análise de ações em tempo real com previsões usando LSTM
"""

import numpy as np
import pandas as pd
import yfinance as yf
import pandas_ta as ta
from datetime import datetime, timedelta
from src.data_preprocessing import DataPreprocessor
from src.models.lstm_model import build_lstm_model
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StockAnalyzer:
    """Análise de ações com previsões LSTM"""
    
    def __init__(self):
        self.preprocessor = DataPreprocessor()
        logger.info("✓ StockAnalyzer initialized")
    
    def fetch_stock_data(self, ticker, period='1y', interval='1d'):
        """
        Baixar dados históricos de uma ação
        
        Args:
            ticker (str): Símbolo da ação (ex: 'AAPL', 'PETR4.SA')
            period (str): Período ('1d', '5d', '1mo', '3mo', '6mo', '1y', etc)
            interval (str): Intervalo ('1m', '5m', '15m', '30m', '60m', '1d', '1wk', '1mo')
            
        Returns:
            pd.DataFrame: Dados históricos com OHLC
        """
        try:
            logger.info(f"📥 Baixando dados de {ticker}...")
            data = yf.download(ticker, period=period, interval=interval, progress=False)
            
            if data is None or data.empty:
                logger.error(f"❌ Nenhum dado encontrado para {ticker}")
                return None
            
            logger.info(f"✓ Dados baixados: {len(data)} registros")
            return data
        except Exception as e:
            logger.error(f"❌ Erro ao baixar dados: {e}")
            return None
    
    def add_technical_indicators(self, df):
        """
        Adicionar indicadores técnicos ao dataframe
        
        Args:
            df (pd.DataFrame): Dataframe com dados OHLCV
            
        Returns:
            pd.DataFrame: Dataframe com indicadores técnicos
        """
        logger.info("📊 Adicionando indicadores técnicos...")
        
        # Flatten MultiIndex columns if present (from yfinance)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0] for col in df.columns]
        
        # Use simple calculations instead of pandas_ta
        
        # RSI - Relative Strength Index (manual calculation)
        try:
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['RSI'] = 100 - (100 / (1 + rs))
        except Exception as e:
            logger.warning(f"Could not compute RSI: {e}")
            df['RSI'] = np.nan
        
        # MACD (manual calculation)
        try:
            ema_12 = df['Close'].ewm(span=12, adjust=False).mean()
            ema_26 = df['Close'].ewm(span=26, adjust=False).mean()
            df['MACD'] = ema_12 - ema_26
            df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
            df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
        except Exception as e:
            logger.warning(f"Could not compute MACD: {e}")
            df['MACD'] = np.nan
            df['MACD_Signal'] = np.nan
            df['MACD_Hist'] = np.nan
        
        # Bollinger Bands (manual calculation)
        try:
            sma20 = df['Close'].rolling(window=20).mean()
            std20 = df['Close'].rolling(window=20).std()
            df['BB_Upper'] = sma20 + (std20 * 2)
            df['BB_Middle'] = sma20
            df['BB_Lower'] = sma20 - (std20 * 2)
        except Exception as e:
            logger.warning(f"Could not compute Bollinger Bands: {e}")
            df['BB_Upper'] = df['Close'] * 1.02
            df['BB_Middle'] = df['Close']
            df['BB_Lower'] = df['Close'] * 0.98
        
        # Médias móveis
        try:
            df['SMA_20'] = df['Close'].rolling(window=20).mean()
            df['SMA_50'] = df['Close'].rolling(window=50).mean()
            df['SMA_200'] = df['Close'].rolling(window=200).mean()
        except Exception as e:
            logger.warning(f"Could not compute SMAs: {e}")
            df['SMA_20'] = df['Close'].rolling(20).mean()
            df['SMA_50'] = df['Close'].rolling(50).mean()
            df['SMA_200'] = df['Close'].rolling(200).mean()
        
        # Remover NaN
        df = df.dropna()
        
        logger.info(f"✓ Adicionados indicadores técnicos")
        return df
    
    def calculate_signal(self, df):
        """
        Calcular sinal de compra/venda baseado em indicadores
        
        Args:
            df (pd.DataFrame): Dataframe com indicadores
            
        Returns:
            str: 'BUY', 'SELL', or 'HOLD'
        """
        if df.empty or len(df) < 1:
            return 'HOLD'
        
        # Flatten MultiIndex columns if present
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0] for col in df.columns]
        
        latest = df.iloc[-1]
        score = 0
        
        # RSI (oversold < 30, overbought > 70)
        if 'RSI' in df.columns and pd.notna(latest['RSI']):
            if latest['RSI'] < 30:
                score += 2
            elif latest['RSI'] > 70:
                score -= 2
        
        # MACD
        if 'MACD' in df.columns and 'MACD_Signal' in df.columns:
            if pd.notna(latest['MACD']) and pd.notna(latest['MACD_Signal']):
                if latest['MACD'] > latest['MACD_Signal']:
                    score += 1
                else:
                    score -= 1
        
        # Preço vs Bollinger Bands
        if 'BB_Lower' in df.columns and 'BB_Upper' in df.columns and 'Close' in df.columns:
            if pd.notna(latest['Close']) and pd.notna(latest['BB_Lower']) and pd.notna(latest['BB_Upper']):
                if latest['Close'] < latest['BB_Lower']:
                    score += 1
                elif latest['Close'] > latest['BB_Upper']:
                    score -= 1
        
        # Médias móveis
        if 'SMA_20' in df.columns and 'SMA_50' in df.columns and 'SMA_200' in df.columns:
            if all(pd.notna(v) for v in [latest['Close'], latest['SMA_20'], latest['SMA_50'], latest['SMA_200']]):
                if latest['Close'] > latest['SMA_20'] > latest['SMA_50'] > latest['SMA_200']:
                    score += 2
                elif latest['Close'] < latest['SMA_20'] < latest['SMA_50'] < latest['SMA_200']:
                    score -= 2
        
        if score >= 2:
            return 'BUY'
        elif score <= -2:
            return 'SELL'
        else:
            return 'HOLD'
    
    def calculate_metrics(self, df):
        """
        Calcular métricas da ação
        
        Args:
            df (pd.DataFrame): Dataframe com dados históricos
            
        Returns:
            dict: Métricas calculadas
        """
        if df.empty:
            return {}
        
        # Flatten MultiIndex columns if present
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0] for col in df.columns]
        
        close_prices = df['Close'].values
        volumes = df['Volume'].values if 'Volume' in df.columns else None
        
        metrics = {
            'Current Price': close_prices[-1],
            'Previous Close': close_prices[-2] if len(close_prices) > 1 else close_prices[-1],
            'Day Change %': ((close_prices[-1] - close_prices[-2]) / close_prices[-2] * 100) if len(close_prices) > 1 else 0,
            '52 Week High': close_prices.max(),
            '52 Week Low': close_prices.min(),
        }
        
        if volumes is not None:
            metrics['Average Daily Volume'] = volumes.mean()
            metrics['30-Day Avg Volume'] = volumes[-30:].mean() if len(volumes) > 30 else volumes.mean()
        
        # Performance
        if len(close_prices) > 30:
            metrics['30-Day Return %'] = ((close_prices[-1] - close_prices[-30]) / close_prices[-30] * 100)
        if len(close_prices) > 90:
            metrics['90-Day Return %'] = ((close_prices[-1] - close_prices[-90]) / close_prices[-90] * 100)
        
        return metrics
    
    def prepare_lstm_data(self, df, lookback=60):
        """
        Preparar dados para previsão LSTM
        
        Args:
            df (pd.DataFrame): Dataframe com indicadores
            lookback (int): Número de dias anteriores
            
        Returns:
            tuple: (X, scaler, dates)
        """
        logger.info("🔧 Preparando dados para LSTM...")
        
        # Usar apenas colunas numéricas relevantes
        numeric_cols = ['Close', 'Volume', 'RSI', 'MACD', 'SMA_20', 'SMA_50']
        available_cols = [col for col in numeric_cols if col in df.columns]
        
        data = df[available_cols].values
        
        # Normalizar
        X_norm, _, _ = self.preprocessor.normalize_data(data, None, None)
        
        # Criar sequências
        X_seq, _ = self.preprocessor.create_sequences(X_norm, lookback=lookback)
        
        logger.info(f"✓ Dados preparados: X.shape={X_seq.shape}")
        return X_seq, self.preprocessor.scaler, df.index[-len(X_seq):]
    
    def predict_next_days(self, df, days=5, lookback=60):
        """
        Prever preço para os próximos dias
        
        Args:
            df (pd.DataFrame): Dataframe com indicadores
            days (int): Número de dias para prever
            lookback (int): Lookback window
            
        Returns:
            dict: Previsões
        """
        logger.info(f"🔮 Prevendo próximos {days} dias...")
        
        try:
            X_seq, scaler, dates = self.prepare_lstm_data(df, lookback)
            
            if X_seq.shape[0] < 1:
                logger.error("❌ Dados insuficientes para LSTM")
                return None
            
            # Construir modelo
            model = build_lstm_model((X_seq.shape[1], X_seq.shape[2]))
            
            # Treinar rapidamente (transfer learning simulation)
            model.fit(X_seq, X_seq, epochs=3, batch_size=16, verbose=0)
            
            # Fazer previsões
            predictions = model.predict(X_seq[-5:], verbose=0)
            
            logger.info(f"✓ Previsão realizada")
            
            return {
                'predictions': predictions,
                'last_price': df['Close'].iloc[-1],
                'model': model
            }
        except Exception as e:
            logger.error(f"❌ Erro na previsão: {e}")
            return None
    
    def get_stock_recommendations(self, tickers, period='6mo'):
        """
        Analisar múltiplas ações e gerar recomendações
        
        Args:
            tickers (list): Lista de símbolos de ações
            period (str): Período para análise
            
        Returns:
            pd.DataFrame: Recomendações
        """
        logger.info(f"📈 Analisando {len(tickers)} ações...")
        
        recommendations = []
        
        for ticker in tickers:
            try:
                df = self.fetch_stock_data(ticker, period=period)
                if df is None:
                    continue
                
                df = self.add_technical_indicators(df)
                signal = self.calculate_signal(df)
                metrics = self.calculate_metrics(df)
                
                recommendations.append({
                    'Ticker': ticker,
                    'Signal': signal,
                    'Current Price': f"${metrics.get('Current Price', 0):.2f}",
                    'Day Change': f"{metrics.get('Day Change %', 0):.2f}%",
                    'RSI': f"{df['RSI'].iloc[-1]:.1f}",
                    'Confidence': 'High' if signal in ['BUY', 'SELL'] else 'Medium'
                })
            except Exception as e:
                logger.error(f"❌ Erro ao analisar {ticker}: {e}")
                continue
        
        df_rec = pd.DataFrame(recommendations)
        logger.info(f"✓ Análise completa: {len(df_rec)} ações")
        return df_rec


def get_brazilian_stocks_list():
    """Retorna lista de ações brasileiras populares"""
    return [
        'PETR4.SA',  # Petrobras
        'VALE3.SA',  # Vale
        'ITUB4.SA',  # Itaú
        'BBDC4.SA',  # Bradesco
        'ABEV3.SA',  # Ambev
        'WEGE3.SA',  # WEG
        'JBSS3.SA',  # JBS
        'RAIL3.SA',  # Rumo
        'LREN3.SA',  # Lojas Renner
        'MGLU3.SA',  # Magazine Luiza
    ]


def get_us_stocks_list():
    """Retorna lista de ações americanas populares"""
    return [
        'AAPL',   # Apple
        'MSFT',   # Microsoft
        'GOOGL',  # Google
        'AMZN',   # Amazon
        'TSLA',   # Tesla
        'META',   # Meta
        'NVDA',   # Nvidia
        'JPM',    # JPMorgan
        'V',      # Visa
        'JNJ',    # Johnson & Johnson
    ]

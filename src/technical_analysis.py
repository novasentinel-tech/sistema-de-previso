"""
Technical Analysis Indicators for Real-Time Data
Computes RSI, MACD, Bollinger Bands, Moving Averages, etc.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from scipy import stats
import logging

logger = logging.getLogger(__name__)


class TechnicalAnalysisEngine:
    """Complete technical analysis engine"""

    @staticmethod
    def calculate_rsi(data: np.ndarray, period: int = 14) -> Dict[str, Any]:
        """Calculate Relative Strength Index (RSI)"""
        try:
            if len(data) < period:
                return {"values": [], "interpretation": "Insufficient data"}
            
            deltas = np.diff(data)
            seed = deltas[:period + 1]
            up = seed[seed >= 0].sum() / period
            down = -seed[seed < 0].sum() / period
            
            rs = up / down if down != 0 else 0
            rsi = np.zeros_like(data, dtype=np.float64)
            rsi[:period] = 100. - 100. / (1. + rs)
            
            for i in range(period, len(data)):
                delta = deltas[i - 1]
                if delta > 0:
                    upval = delta
                    downval = 0.
                else:
                    upval = 0.
                    downval = -delta
                
                up = (up * (period - 1) + upval) / period
                down = (down * (period - 1) + downval) / period
                
                rs = up / down if down != 0 else 0
                rsi[i] = 100. - 100. / (1. + rs)
            
            return {
                "values": rsi.tolist(),
                "current": float(rsi[-1]),
                "overbought": float(rsi[-1]) > 70,
                "oversold": float(rsi[-1]) < 30,
                "interpretation": "Overbought" if float(rsi[-1]) > 70 else ("Oversold" if float(rsi[-1]) < 30 else "Neutral"),
                "threshold_overbought": 70,
                "threshold_oversold": 30
            }
        except Exception as e:
            logger.error(f"RSI calculation error: {str(e)}")
            return {"values": [], "error": str(e)}

    @staticmethod
    def calculate_macd(data: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, Any]:
        """Calculate MACD (Moving Average Convergence Divergence)"""
        try:
            if len(data) < slow:
                return {"error": "Insufficient data"}
            
            ema_fast = TechnicalAnalysisEngine._ema(data, fast)
            ema_slow = TechnicalAnalysisEngine._ema(data, slow)
            
            macd_line = ema_fast - ema_slow
            signal_line = TechnicalAnalysisEngine._ema(macd_line, signal)
            histogram = macd_line - signal_line
            
            # Determine cross signal
            if len(macd_line) >= 2:
                prev_histogram = histogram[-2]
                curr_histogram = histogram[-1]
                if prev_histogram < 0 and curr_histogram > 0:
                    signal_cross = "bullish"
                elif prev_histogram > 0 and curr_histogram < 0:
                    signal_cross = "bearish"
                else:
                    signal_cross = "neutral"
            else:
                signal_cross = "neutral"
            
            return {
                "macd_line": macd_line.tolist(),
                "signal_line": signal_line.tolist(),
                "histogram": histogram.tolist(),
                "current_macd": float(macd_line[-1]),
                "current_signal": float(signal_line[-1]),
                "current_histogram": float(histogram[-1]),
                "signal_cross": signal_cross
            }
        except Exception as e:
            logger.error(f"MACD calculation error: {str(e)}")
            return {"error": str(e)}

    @staticmethod
    def calculate_bollinger_bands(data: np.ndarray, period: int = 20, num_std: float = 2.0) -> Dict[str, Any]:
        """Calculate Bollinger Bands"""
        try:
            if len(data) < period:
                return {"error": "Insufficient data"}
            
            sma = TechnicalAnalysisEngine._sma(data, period)
            std = np.array([np.std(data[max(0, i - period + 1):i + 1]) for i in range(len(data))])
            
            upper_band = sma + (std * num_std)
            lower_band = sma - (std * num_std)
            
            # Band width
            band_width = (upper_band[-1] - lower_band[-1]) / sma[-1] if sma[-1] != 0 else 0
            
            # Price position (0 = lower band, 1 = upper band)
            if upper_band[-1] != lower_band[-1]:
                position = (data[-1] - lower_band[-1]) / (upper_band[-1] - lower_band[-1])
            else:
                position = 0.5
            
            return {
                "upper": upper_band.tolist(),
                "middle": sma.tolist(),
                "lower": lower_band.tolist(),
                "current_upper": float(upper_band[-1]),
                "current_middle": float(sma[-1]),
                "current_lower": float(lower_band[-1]),
                "band_width": float(band_width),
                "price_position": float(np.clip(position, 0, 1)),
                "interpretation": "Overbought" if position > 0.9 else ("Oversold" if position < 0.1 else "Neutral")
            }
        except Exception as e:
            logger.error(f"Bollinger Bands calculation error: {str(e)}")
            return {"error": str(e)}

    @staticmethod
    def calculate_moving_averages(data: np.ndarray, periods: List[int] = [10, 20, 50]) -> Dict[str, Any]:
        """Calculate Simple and Exponential Moving Averages"""
        try:
            result = {}
            
            # SMA
            for period in periods:
                if len(data) >= period:
                    sma = TechnicalAnalysisEngine._sma(data, period)
                    result[f"sma_{period}"] = float(sma[-1])
                    result[f"sma_{period}_array"] = sma.tolist()
            
            # EMA
            for period in periods:
                if len(data) >= period:
                    ema = TechnicalAnalysisEngine._ema(data, period)
                    result[f"ema_{period}"] = float(ema[-1])
                    result[f"ema_{period}_array"] = ema.tolist()
            
            return result
        except Exception as e:
            logger.error(f"Moving averages calculation error: {str(e)}")
            return {"error": str(e)}

    @staticmethod
    def calculate_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> Dict[str, Any]:
        """Calculate Average True Range (ATR)"""
        try:
            if len(high) < period or len(low) < period or len(close) < period:
                return {"error": "Insufficient data"}
            
            tr1 = high - low
            tr2 = np.abs(high - close[0])
            tr3 = np.abs(low - close[0])
            
            tr = np.maximum(np.maximum(tr1, tr2), tr3)
            atr = TechnicalAnalysisEngine._sma(tr, period)
            
            return {
                "values": atr.tolist(),
                "current": float(atr[-1]),
                "volatility": float(atr[-1] / close[-1]) if close[-1] != 0 else 0
            }
        except Exception as e:
            logger.error(f"ATR calculation error: {str(e)}")
            return {"error": str(e)}

    @staticmethod
    def calculate_stochastic(high: np.ndarray, low: np.ndarray, close: np.ndarray, 
                            k_period: int = 14, d_period: int = 3) -> Dict[str, Any]:
        """Calculate Stochastic Oscillator"""
        try:
            if len(high) < k_period or len(low) < k_period or len(close) < k_period:
                return {"error": "Insufficient data"}
            
            lowest_low = np.array([np.min(low[max(0, i - k_period + 1):i + 1]) for i in range(len(low))])
            highest_high = np.array([np.max(high[max(0, i - k_period + 1):i + 1]) for i in range(len(high))])
            
            k_percent = 100 * (close - lowest_low) / (highest_high - lowest_low + 1e-10)
            d_percent = TechnicalAnalysisEngine._sma(k_percent, d_period)
            
            return {
                "k_percent": k_percent.tolist(),
                "d_percent": d_percent.tolist(),
                "current_k": float(k_percent[-1]),
                "current_d": float(d_percent[-1]),
                "overbought": float(k_percent[-1]) > 80,
                "oversold": float(k_percent[-1]) < 20
            }
        except Exception as e:
            logger.error(f"Stochastic calculation error: {str(e)}")
            return {"error": str(e)}

    @staticmethod
    def calculate_trend_analysis(data: np.ndarray) -> Dict[str, Any]:
        """Analyze trend direction and strength"""
        try:
            if len(data) < 3:
                return {"error": "Insufficient data"}
            
            # Linear regression
            x = np.arange(len(data))
            slope, intercept = np.polyfit(x, data, 1)
            
            # Trend classification
            if slope > 0.01:
                trend = "upward"
            elif slope < -0.01:
                trend = "downward"
            else:
                trend = "sideways"
            
            # Trend strength (R²)
            y_pred = slope * x + intercept
            ss_res = np.sum((data - y_pred) ** 2)
            ss_tot = np.sum((data - np.mean(data)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
            
            # Change percent
            change_percent = ((data[-1] - data[0]) / data[0] * 100) if data[0] != 0 else 0
            
            return {
                "overall_trend": trend,
                "trend_strength": float(r_squared),
                "slope": float(slope),
                "change_percent": float(change_percent),
                "volatility": float(np.std(np.diff(data) / data[:-1])) if len(data) > 1 else 0
            }
        except Exception as e:
            logger.error(f"Trend analysis error: {str(e)}")
            return {"error": str(e)}

    @staticmethod
    def detect_anomalies(data: np.ndarray, threshold: float = 2.5) -> Dict[str, Any]:
        """Detect anomalies using Z-score"""
        try:
            # Convert to 1D array and ensure it's numeric
            data_flat = np.asarray(data, dtype=np.float64).flatten().tolist()
            # Use scipy.stats.zscore with proper conversion
            z_scores = np.abs(stats.zscore(data_flat.tolist()))
            anomalies = np.where(z_scores > threshold)[0]
            
            anomaly_list = []
            for idx in anomalies:
                anomaly_list.append({
                    "period": int(idx),
                    "value": float(data_flat[idx]),
                    "zscore": float(z_scores[idx]),
                    "anomaly_type": "positive_spike" if data_flat[idx] > np.mean(data_flat) else "negative_dip"
                })
            
            return {
                "detected": len(anomalies) > 0,
                "count": int(len(anomalies)),
                "anomalies": anomaly_list,
                "threshold": threshold
            }
        except Exception as e:
            logger.error(f"Anomaly detection error: {str(e)}")
            return {"error": str(e)}

    @staticmethod
    def calculate_statistics(data: np.ndarray) -> Dict[str, Any]:
        """Calculate comprehensive statistics"""
        try:
            return {
                "mean": float(np.mean(data)),
                "std": float(np.std(data)),
                "min": float(np.min(data)),
                "max": float(np.max(data)),
                "median": float(np.median(data)),
                "percentile_25": float(np.percentile(data, 25)),
                "percentile_75": float(np.percentile(data, 75)),
                "skewness": float(stats.skew(data)),
                "kurtosis": float(stats.kurtosis(data)),
                "range": float(np.max(data) - np.min(data)),
                "variance": float(np.var(data))
            }
        except Exception as e:
            logger.error(f"Statistics calculation error: {str(e)}")
            return {"error": str(e)}

    @staticmethod
    def calculate_correlations(forecast: np.ndarray, actual: np.ndarray, 
                              volume: Optional[np.ndarray] = None, rsi: Optional[np.ndarray] = None, 
                              macd: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Calculate correlations between different indicators"""
        try:
            result = {}
            
            # Ensure same length
            min_len = min(len(forecast), len(actual))
            forecast_trunc = forecast[:min_len]
            actual_trunc = actual[:min_len]
            
            if np.std(forecast_trunc) > 0 and np.std(actual_trunc) > 0:
                result["forecast_vs_actual"] = float(np.corrcoef(forecast_trunc, actual_trunc)[0, 1])
            
            if volume is not None and len(volume) >= min_len:
                volume_trunc = volume[:min_len]
                if np.std(volume_trunc) > 0 and np.std(forecast_trunc) > 0:
                    result["forecast_vs_volume"] = float(np.corrcoef(forecast_trunc, volume_trunc)[0, 1])
            
            if rsi is not None and len(rsi) >= min_len:
                rsi_trunc = rsi[:min_len]
                if np.std(rsi_trunc) > 0 and np.std(forecast_trunc) > 0:
                    result["forecast_vs_rsi"] = float(np.corrcoef(forecast_trunc, rsi_trunc)[0, 1])
            
            if macd is not None and len(macd) >= min_len:
                macd_trunc = macd[:min_len]
                if np.std(macd_trunc) > 0 and np.std(forecast_trunc) > 0:
                    result["forecast_vs_macd"] = float(np.corrcoef(forecast_trunc, macd_trunc)[0, 1])
            
            return result
        except Exception as e:
            logger.error(f"Correlation calculation error: {str(e)}")
            return {}

    @staticmethod
    def calculate_confidence_intervals(forecast: np.ndarray, std_error: np.ndarray, 
                                      confidence_level: float = 0.95) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate confidence intervals"""
        try:
            from scipy.stats import t
            
            # Use t-distribution
            df = len(forecast) - 1
            t_val = t.ppf((1 + confidence_level) / 2, df)
            
            margin_of_error = t_val * std_error
            
            lower = forecast - margin_of_error
            upper = forecast + margin_of_error
            
            return lower, upper
        except Exception as e:
            logger.error(f"Confidence interval calculation error: {str(e)}")
            return np.array([]), np.array([])

    @staticmethod
    def calculate_directional_accuracy(actual: np.ndarray, forecast: np.ndarray) -> float:
        """Calculate percentage of correct direction predictions"""
        try:
            if len(actual) < 2 or len(forecast) < 2:
                return 0.0
            
            actual_direction = np.diff(actual) > 0
            forecast_direction = np.diff(forecast) > 0
            
            correct = np.sum(actual_direction == forecast_direction)
            accuracy = correct / len(actual_direction) if len(actual_direction) > 0 else 0
            
            return float(accuracy)
        except Exception as e:
            logger.error(f"Directional accuracy calculation error: {str(e)}")
            return 0.0

    # Helper methods
    @staticmethod
    def _sma(data: np.ndarray, period: int) -> np.ndarray:
        """Simple Moving Average"""
        result = pd.Series(data).rolling(window=period).mean().values
        return np.asarray(result, dtype=np.float64)

    @staticmethod
    def _ema(data: np.ndarray, period: int) -> np.ndarray:
        """Exponential Moving Average"""
        result = pd.Series(data).ewm(span=period, adjust=False).mean().values
        return np.asarray(result, dtype=np.float64)


def generate_signals(technical_data: Dict[str, Any]) -> Dict[str, Any]:
    """Generate trading signals based on technical indicators"""
    try:
        buy_signals = 0
        sell_signals = 0
        buy_strength = 0
        sell_strength = 0
        
        # RSI signals
        if "rsi" in technical_data:
            if technical_data["rsi"].get("oversold"):
                buy_signals += 1
                buy_strength += 0.3
            elif technical_data["rsi"].get("overbought"):
                sell_signals += 1
                sell_strength += 0.3
        
        # MACD signals
        if "macd" in technical_data:
            if technical_data["macd"].get("signal_cross") == "bullish":
                buy_signals += 1
                buy_strength += 0.4
            elif technical_data["macd"].get("signal_cross") == "bearish":
                sell_signals += 1
                sell_strength += 0.4
        
        # Bollinger Bands signals
        if "bollinger_bands" in technical_data:
            position = technical_data["bollinger_bands"].get("price_position", 0.5)
            if position < 0.2:
                buy_signals += 1
                buy_strength += 0.3
            elif position > 0.8:
                sell_signals += 1
                sell_strength += 0.3
        
        # Determine overall signal
        if buy_signals > sell_signals:
            overall_signal = "BUY"
            confidence = min(buy_strength, 1.0)
        elif sell_signals > buy_signals:
            overall_signal = "SELL"
            confidence = min(sell_strength, 1.0)
        else:
            overall_signal = "HOLD"
            confidence = 0.5
        
        # Map to category
        if overall_signal == "BUY":
            recommendation = "STRONG_BUY" if confidence > 0.8 else "BUY"
        elif overall_signal == "SELL":
            recommendation = "STRONG_SELL" if confidence > 0.8 else "SELL"
        else:
            recommendation = "HOLD"
        
        return {
            "buy_signals": buy_signals,
            "sell_signals": sell_signals,
            "overall_signal": overall_signal,
            "recommendation": recommendation,
            "confidence": float(confidence),
            "signal_sources": []
        }
    except Exception as e:
        logger.error(f"Signal generation error: {str(e)}")
        return {"error": str(e), "overall_signal": "HOLD"}

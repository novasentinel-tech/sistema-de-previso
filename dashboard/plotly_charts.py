"""
Plotly Visualization Module
Creates interactive charts for time series and predictions
"""

import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_time_series_plot(df, columns, title='Time Series'):
    """
    Create interactive time series plot
    
    Args:
        df (pd.DataFrame): Input dataframe with datetime index
        columns (list): Column names to plot
        title (str): Plot title
        
    Returns:
        plotly.graph_objects.Figure: Interactive plot
        
    Example:
        >>> fig = create_time_series_plot(df, ['temperature', 'humidity'])
        >>> fig.show()
    """
    
    fig = go.Figure()
    
    if not isinstance(columns, list):
        columns = [columns]
    
    colors = px.colors.qualitative.Set1
    
    for idx, col in enumerate(columns):
        color = colors[idx % len(colors)]
        
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df[col],
            mode='lines',
            name=col,
            line=dict(color=color, width=2),
            hovertemplate=f'<b>{col}</b><br>Date: %{{x | %Y-%m-%d %H:%M}}<br>Value: %{{y:.4f}}<extra></extra>'
        ))
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=20)),
        xaxis_title='Time',
        yaxis_title='Value',
        template='plotly_dark',
        hovermode='x unified',
        height=600,
        showlegend=True,
        legend=dict(x=0.01, y=0.99)
    )
    
    logger.info(f"✓ Created time series plot for {len(columns)} columns")
    return fig


def create_forecast_plot(historical_df, forecast_df, col_name, title=None):
    """
    Create forecast vs historical plot
    
    Args:
        historical_df (pd.DataFrame): Historical data
        forecast_df (pd.DataFrame): Forecast data with columns: 'yhat', 'yhat_lower', 'yhat_upper'
        col_name (str): Column name
        title (str): Plot title
        
    Returns:
        plotly.graph_objects.Figure: Interactive plot
    """
    
    if title is None:
        title = f'{col_name} - Forecast vs Historical'
    
    fig = go.Figure()
    
    # Add historical data
    fig.add_trace(go.Scatter(
        x=historical_df.index,
        y=historical_df[col_name],
        mode='lines',
        name='Historical',
        line=dict(color='blue', width=2),
        hovertemplate='<b>Historical</b><br>Date: %{x | %Y-%m-%d}<br>Value: %{y:.4f}<extra></extra>'
    ))
    
    # Add forecast
    fig.add_trace(go.Scatter(
        x=forecast_df['ds'],
        y=forecast_df['yhat'],
        mode='lines',
        name='Forecast',
        line=dict(color='red', width=2, dash='dash'),
        hovertemplate='<b>Forecast</b><br>Date: %{x | %Y-%m-%d}<br>Value: %{y:.4f}<extra></extra>'
    ))
    
    # Add confidence interval
    fig.add_trace(go.Scatter(
        x=forecast_df['ds'],
        y=forecast_df['yhat_upper'],
        fill=None,
        mode='lines',
        line_color='rgba(0,0,0,0)',
        showlegend=False,
        hoverinfo='skip'
    ))
    
    fig.add_trace(go.Scatter(
        x=forecast_df['ds'],
        y=forecast_df['yhat_lower'],
        fill='tonexty',
        mode='lines',
        line_color='rgba(0,0,0,0)',
        name='95% Confidence Interval',
        fillcolor='rgba(255,0,0,0.2)',
        hovertemplate='<b>CI</b><br>Date: %{x | %Y-%m-%d}<br>Range: %{y:.4f}<extra></extra>'
    ))
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=20)),
        xaxis_title='Time',
        yaxis_title='Value',
        template='plotly_dark',
        hovermode='x unified',
        height=600,
        showlegend=True,
        legend=dict(x=0.01, y=0.99)
    )
    
    logger.info(f"✓ Created forecast plot for '{col_name}'")
    return fig


def create_prediction_vs_actual(y_true, y_pred, title='Predictions vs Actual'):
    """
    Create interactive prediction vs actual plot
    
    Args:
        y_true (np.ndarray): True values
        y_pred (np.ndarray): Predicted values
        title (str): Plot title
        
    Returns:
        plotly.graph_objects.Figure: Interactive plot
    """
    
    y_true = y_true.flatten() if isinstance(y_true, np.ndarray) else y_true.values
    y_pred = y_pred.flatten() if isinstance(y_pred, np.ndarray) else y_pred.values
    
    fig = go.Figure()
    
    timesteps = np.arange(len(y_true))
    
    fig.add_trace(go.Scatter(
        x=timesteps,
        y=y_true,
        mode='lines',
        name='Actual',
        line=dict(color='blue', width=2),
        hovertemplate='<b>Actual</b><br>Step: %{x}<br>Value: %{y:.4f}<extra></extra>'
    ))
    
    fig.add_trace(go.Scatter(
        x=timesteps,
        y=y_pred,
        mode='lines',
        name='Predicted',
        line=dict(color='red', width=2, dash='dash'),
        hovertemplate='<b>Predicted</b><br>Step: %{x}<br>Value: %{y:.4f}<extra></extra>'
    ))
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=20)),
        xaxis_title='Time Step',
        yaxis_title='Value',
        template='plotly_dark',
        hovermode='x unified',
        height=600,
        showlegend=True
    )
    
    logger.info("✓ Created prediction vs actual plot")
    return fig


def create_error_heatmap(df_errors, title='Prediction Errors Heatmap'):
    """
    Create heatmap of prediction errors
    
    Args:
        df_errors (pd.DataFrame): Dataframe with errors
        title (str): Plot title
        
    Returns:
        plotly.graph_objects.Figure: Interactive heatmap
    """
    
    fig = go.Figure(data=go.Heatmap(
        z=df_errors.values,
        x=df_errors.columns,
        y=df_errors.index,
        colorscale='RdBu',
        zmid=0,
        colorbar=dict(title='Error')
    ))
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=20)),
        xaxis_title='Variables',
        yaxis_title='Time Step',
        template='plotly_dark',
        height=600
    )
    
    logger.info("✓ Created error heatmap")
    return fig


def create_metrics_comparison(metrics_dict, title='Model Metrics Comparison'):
    """
    Create bar chart comparing metrics
    
    Args:
        metrics_dict (dict): Dictionary of metrics {metric_name: value}
        title (str): Plot title
        
    Returns:
        plotly.graph_objects.Figure: Interactive bar chart
    """
    
    fig = go.Figure(data=go.Bar(
        x=list(metrics_dict.keys()),
        y=list(metrics_dict.values()),
        marker=dict(color=list(metrics_dict.values()), 
                   colorscale='Viridis',
                   showscale=True),
        text=[f'{v:.4f}' for v in metrics_dict.values()],
        textposition='auto'
    ))
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=20)),
        xaxis_title='Metrics',
        yaxis_title='Value',
        template='plotly_dark',
        height=500,
        showlegend=False
    )
    
    logger.info("✓ Created metrics comparison chart")
    return fig


def create_residuals_plot(residuals, title='Residuals Analysis'):
    """
    Create interactive residuals plot
    
    Args:
        residuals (np.ndarray): Residuals
        title (str): Plot title
        
    Returns:
        plotly.graph_objects.Figure: Interactive scatter plot
    """
    
    residuals = residuals.flatten() if isinstance(residuals, np.ndarray) else residuals.values
    
    fig = go.Figure()
    
    timesteps = np.arange(len(residuals))
    
    fig.add_trace(go.Scatter(
        x=timesteps,
        y=residuals,
        mode='markers',
        name='Residuals',
        marker=dict(size=6, color='steelblue', opacity=0.7),
        hovertemplate='<b>Residual</b><br>Step: %{x}<br>Value: %{y:.4f}<extra></extra>'
    ))
    
    # Add reference line at 0
    fig.add_hline(y=0, line_dash='dash', line_color='red', annotation_text='Zero Error')
    
    # Add ±2 std dev lines
    std_val = residuals.std()
    fig.add_hline(y=2*std_val, line_dash='dot', line_color='orange', opacity=0.5)
    fig.add_hline(y=-2*std_val, line_dash='dot', line_color='orange', opacity=0.5)
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=20)),
        xaxis_title='Time Step',
        yaxis_title='Residual',
        template='plotly_dark',
        height=600
    )
    
    logger.info("✓ Created residuals plot")
    return fig


def create_distribution_plot(values, name='Distribution', title=None):
    """
    Create distribution histogram
    
    Args:
        values (np.ndarray): Values to plot
        name (str): Variable name
        title (str): Plot title
        
    Returns:
        plotly.graph_objects.Figure: Interactive histogram
    """
    
    if title is None:
        title = f'{name} Distribution'
    
    values = values.flatten() if isinstance(values, np.ndarray) else values.values
    
    fig = go.Figure(data=go.Histogram(
        x=values,
        name=name,
        nbinsx=30,
        marker=dict(color='steelblue', opacity=0.8)
    ))
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=20)),
        xaxis_title='Value',
        yaxis_title='Frequency',
        template='plotly_dark',
        height=500
    )
    
    logger.info(f"✓ Created distribution plot for '{name}'")
    return fig


if __name__ == "__main__":
    print("✓ Plotly charts module ready!")
    print("\nUsage example:")
    print("  from dashboard.plotly_charts import create_time_series_plot")
    print("  fig = create_time_series_plot(df, ['col1', 'col2'])")
    print("  fig.show()")

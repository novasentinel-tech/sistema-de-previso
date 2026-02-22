"""Dashboard module for TOTEM_DEEPSEA"""

from .plotly_charts import (
    create_time_series_plot,
    create_forecast_plot,
    create_prediction_vs_actual
)

__all__ = [
    'create_time_series_plot',
    'create_forecast_plot',
    'create_prediction_vs_actual'
]

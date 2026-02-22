"""
Model Evaluation Module
Calculates metrics and generates evaluation plots
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, r2_score
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set style for plots
sns.set_style("darkgrid")
plt.rcParams['figure.figsize'] = (14, 6)


def calculate_metrics(y_true, y_pred, metric_names=None):
    """
    Calculate evaluation metrics
    
    Args:
        y_true (np.ndarray): True values
        y_pred (np.ndarray): Predicted values
        metric_names (list): List of metrics to calculate
        
    Returns:
        dict: Dictionary of calculated metrics
        
    Example:
        >>> metrics = calculate_metrics(y_test, y_pred)
        >>> for metric, value in metrics.items():
        ...     print(f"{metric}: {value:.4f}")
    """
    
    if metric_names is None:
        metric_names = ['mae', 'rmse', 'mape', 'r2']
    
    metrics = {}
    
    # Ensure arrays are 1D
    y_true = y_true.flatten() if isinstance(y_true, np.ndarray) else y_true
    y_pred = y_pred.flatten() if isinstance(y_pred, np.ndarray) else y_pred
    
    # Replace NaN and inf with 0
    y_true = np.nan_to_num(y_true, nan=0.0, posinf=0.0, neginf=0.0)
    y_pred = np.nan_to_num(y_pred, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Clip values
    y_true = np.clip(y_true, -1e6, 1e6)
    y_pred = np.clip(y_pred, -1e6, 1e6)
    
    if 'mae' in metric_names:
        metrics['mae'] = mean_absolute_error(y_true, y_pred)
    
    if 'rmse' in metric_names:
        metrics['rmse'] = np.sqrt(mean_squared_error(y_true, y_pred))
    
    if 'mape' in metric_names:
        # Avoid division by zero
        mask = y_true != 0
        if mask.sum() > 0:
            metrics['mape'] = mean_absolute_percentage_error(y_true[mask], y_pred[mask])
        else:
            metrics['mape'] = np.nan
    
    if 'r2' in metric_names:
        metrics['r2'] = r2_score(y_true, y_pred)
    
    if 'mse' in metric_names:
        metrics['mse'] = mean_squared_error(y_true, y_pred)
    
    logger.info("📊 Evaluation Metrics:")
    for metric, value in metrics.items():
        logger.info(f"  {metric.upper()}: {value:.6f}")
    
    return metrics


def plot_predictions(y_true, y_pred, title='Predictions vs Actual', 
                     save_path=None, figsize=(14, 6)):
    """
    Plot predictions vs actual values
    
    Args:
        y_true (np.ndarray): True values
        y_pred (np.ndarray): Predicted values
        title (str): Plot title
        save_path (str): Path to save figure (optional)
        figsize (tuple): Figure size
        
    Returns:
        matplotlib.figure.Figure: Figure object
    """
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Flatten if needed
    y_true = y_true.flatten() if isinstance(y_true, np.ndarray) else y_true
    y_pred = y_pred.flatten() if isinstance(y_pred, np.ndarray) else y_pred
    
    # Plot
    timesteps = np.arange(len(y_true))
    
    ax.plot(timesteps, y_true, 'b-', label='Actual', linewidth=2, alpha=0.8)
    ax.plot(timesteps, y_pred, 'r--', label='Predicted', linewidth=2, alpha=0.8)
    
    ax.set_xlabel('Time Step', fontsize=12)
    ax.set_ylabel('Value', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"✓ Plot saved to: {save_path}")
    
    return fig


def plot_error_distribution(y_true, y_pred, title='Prediction Errors Distribution',
                           save_path=None, figsize=(14, 5)):
    """
    Plot error distribution
    
    Args:
        y_true (np.ndarray): True values
        y_pred (np.ndarray): Predicted values
        title (str): Plot title
        save_path (str): Path to save figure
        figsize (tuple): Figure size
        
    Returns:
        matplotlib.figure.Figure: Figure object
    """
    
    # Flatten if needed
    y_true = y_true.flatten() if isinstance(y_true, np.ndarray) else y_true
    y_pred = y_pred.flatten() if isinstance(y_pred, np.ndarray) else y_pred
    
    errors = y_true - y_pred
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Histogram
    axes[0].hist(errors, bins=30, edgecolor='black', alpha=0.7, color='steelblue')
    axes[0].axvline(errors.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {errors.mean():.4f}')
    axes[0].set_xlabel('Error', fontsize=11)
    axes[0].set_ylabel('Frequency', fontsize=11)
    axes[0].set_title('Error Distribution', fontsize=12, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Q-Q plot
    from scipy import stats
    stats.probplot(errors, dist="norm", plot=axes[1])
    axes[1].set_title('Q-Q Plot', fontsize=12, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"✓ Error plot saved to: {save_path}")
    
    return fig


def plot_residuals(y_true, y_pred, save_path=None, figsize=(14, 5)):
    """
    Plot residuals over time
    
    Args:
        y_true (np.ndarray): True values
        y_pred (np.ndarray): Predicted values
        save_path (str): Path to save figure
        figsize (tuple): Figure size
        
    Returns:
        matplotlib.figure.Figure: Figure object
    """
    
    y_true = y_true.flatten() if isinstance(y_true, np.ndarray) else y_true
    y_pred = y_pred.flatten() if isinstance(y_pred, np.ndarray) else y_pred
    
    residuals = y_true - y_pred
    timesteps = np.arange(len(residuals))
    
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.scatter(timesteps, residuals, alpha=0.6, s=30, color='steelblue')
    ax.axhline(y=0, color='red', linestyle='--', linewidth=2)
    ax.fill_between(timesteps, -2*residuals.std(), 2*residuals.std(), 
                     alpha=0.2, color='orange', label='±2 Std Dev')
    
    ax.set_xlabel('Time Step', fontsize=12)
    ax.set_ylabel('Residual', fontsize=12)
    ax.set_title('Residuals Over Time', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"✓ Residuals plot saved to: {save_path}")
    
    return fig


def compare_models(models_results, figsize=(15, 6)):
    """
    Compare multiple models' metrics
    
    Args:
        models_results (dict): Dictionary with model names as keys and 
                              metrics dicts as values
        figsize (tuple): Figure size
        
    Returns:
        matplotlib.figure.Figure: Figure object
    """
    
    # Prepare data for comparison
    df_comparison = pd.DataFrame(models_results).T
    
    fig, axes = plt.subplots(1, len(df_comparison.columns), figsize=figsize)
    
    if len(df_comparison.columns) == 1:
        axes = [axes]
    
    for idx, metric in enumerate(df_comparison.columns):
        axes[idx].bar(df_comparison.index, df_comparison[metric], alpha=0.7, color='steelblue')
        axes[idx].set_ylabel(metric.upper(), fontsize=11)
        axes[idx].set_title(f'{metric.upper()} Comparison', fontsize=12, fontweight='bold')
        axes[idx].grid(True, alpha=0.3, axis='y')
        plt.setp(axes[idx].xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.suptitle('Model Performance Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    return fig


def create_evaluation_report(y_true, y_pred, model_name='Model', save_path=None):
    """
    Create comprehensive evaluation report
    
    Args:
        y_true (np.ndarray): True values
        y_pred (np.ndarray): Predicted values
        model_name (str): Name of the model
        save_path (str): Path to save report
        
    Returns:
        dict: Evaluation report
    """
    
    logger.info(f"\n{'='*60}")
    logger.info(f"EVALUATION REPORT: {model_name}")
    logger.info(f"{'='*60}")
    
    # Calculate metrics
    metrics = calculate_metrics(y_true, y_pred)
    
    # Calculate additional statistics
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()
    
    report = {
        'model_name': model_name,
        'metrics': metrics,
        'statistics': {
            'y_true_mean': float(y_true_flat.mean()),
            'y_true_std': float(y_true_flat.std()),
            'y_pred_mean': float(y_pred_flat.mean()),
            'y_pred_std': float(y_pred_flat.std()),
            'samples': len(y_true_flat)
        }
    }
    
    if save_path:
        import json
        with open(save_path, 'w') as f:
            json.dump(report, f, indent=4)
        logger.info(f"✓ Report saved to: {save_path}")
    
    logger.info(f"{'='*60}\n")
    
    return report


if __name__ == "__main__":
    print("✓ Evaluation module ready!")
    print("\nUsage example:")
    print("  from evaluation import calculate_metrics, plot_predictions")
    print("  metrics = calculate_metrics(y_test, y_pred)")
    print("  plot_predictions(y_test, y_pred)")

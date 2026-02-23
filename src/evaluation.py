import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')


def calculate_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), 1e-10))) * 100
    r2 = r2_score(y_true, y_pred)
    
    return {
        'MAE': mae,
        'RMSE': rmse,
        'MAPE': mape,
        'R2': r2
    }


def calculate_direction_accuracy(y_true, y_pred):
    true_direction = np.diff(y_true) > 0
    pred_direction = np.diff(y_pred) > 0
    accuracy = np.mean(true_direction == pred_direction) * 100
    return accuracy


def evaluate_model(y_true, y_pred, model_name='Model'):
    metrics = calculate_metrics(y_true, y_pred)
    direction_acc = calculate_direction_accuracy(y_true, y_pred)
    
    results = {
        'Model': model_name,
        'MAE': metrics['MAE'],
        'RMSE': metrics['RMSE'],
        'MAPE': metrics['MAPE'],
        'R2': metrics['R2'],
        'Direction_Accuracy': direction_acc
    }
    
    return results


def plot_predictions(y_true, y_pred, title='Predictions vs Actual', figsize=(12, 6)):
    plt.figure(figsize=figsize)
    plt.plot(y_true, label='Actual', linewidth=2)
    plt.plot(y_pred, label='Predicted', linewidth=2)
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    return plt


def plot_residuals(y_true, y_pred, figsize=(12, 4)):
    residuals = y_true - y_pred
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    axes[0].plot(residuals, marker='o', linestyle='-', linewidth=1, markersize=3)
    axes[0].axhline(y=0, color='r', linestyle='--')
    axes[0].set_xlabel('Time')
    axes[0].set_ylabel('Residual')
    axes[0].set_title('Residuals Over Time')
    axes[0].grid(True, alpha=0.3)
    
    axes[1].hist(residuals, bins=30, edgecolor='black', alpha=0.7)
    axes[1].set_xlabel('Residual')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Residuals Distribution')
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    return fig


def plot_metrics_comparison(results_list, figsize=(12, 6)):
    df = pd.DataFrame(results_list)
    metrics = ['MAE', 'RMSE', 'MAPE', 'R2', 'Direction_Accuracy']
    
    x = np.arange(len(df))
    width = 0.15
    
    fig, ax = plt.subplots(figsize=figsize)
    
    for i, metric in enumerate(metrics):
        ax.bar(x + i*width, df[metric], width, label=metric)
    
    ax.set_xlabel('Model')
    ax.set_ylabel('Score')
    ax.set_title('Model Performance Comparison')
    ax.set_xticks(x + width * 2)
    ax.set_xticklabels(df['Model'])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    return fig


def generate_evaluation_report(y_true, y_pred, model_name='Model'):
    metrics = calculate_metrics(y_true, y_pred)
    direction_acc = calculate_direction_accuracy(y_true, y_pred)
    
    report = f"""
    ====== Relatório de Avaliação ======
    Modelo: {model_name}
    
    Métricas:
    - MAE (Mean Absolute Error): {metrics['MAE']:.4f}
    - RMSE (Root Mean Squared Error): {metrics['RMSE']:.4f}
    - MAPE (Mean Absolute Percentage Error): {metrics['MAPE']:.2f}%
    - R² Score: {metrics['R2']:.4f}
    - Direction Accuracy: {direction_acc:.2f}%
    
    Interpretação:
    - MAE: Erro absoluto médio em unidades reais
    - RMSE: Penaliza erros grandes mais que o MAE
    - MAPE: Percentual de erro relativo
    - R²: Proporção de variância explicada (0-1, maior é melhor)
    - Direction Accuracy: % de acertos na direção da mudança
    """
    
    return report

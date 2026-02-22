#!/usr/bin/env python
"""
System validation script - testes rápidos para verificar se tudo está funcionando
"""

import sys
import os
import numpy as np
import pandas as pd

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.config import RAW_PATH, PROCESSED_PATH
from src.data_preprocessing import DataPreprocessor
from src.models.lstm_model import build_lstm_model, train_lstm
from src.evaluation import calculate_metrics
from src.prediction import PredictionEngine

print("\n" + "="*60)
print("🧪 SISTEMA DE PREVISÃO - TESTE DE VALIDAÇÃO")
print("="*60)

# Test 1: Preprocessor initialization
print("\n1️⃣  Testando DataPreprocessor...")
try:
    preprocessor = DataPreprocessor()
    print("✅ DataPreprocessor funcionando")
except Exception as e:
    print(f"❌ Erro: {e}")
    sys.exit(1)

# Test 2: Create sample data
print("\n2️⃣  Gerando dados de amostra...")
try:
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=200, freq='h')
    df = pd.DataFrame({
        'temperature': 20 + 5*np.sin(np.arange(200)*2*np.pi/24) + np.random.randn(200)*0.5,
        'humidity': 60 + 10*np.sin(np.arange(200)*2*np.pi/24) + np.random.randn(200)*1,
        'pressure': 1013 + np.random.randn(200)*0.5,
    }, index=dates)
    df.index.name = 'timestamp'
    print(f"✅ Dados criados: {df.shape}")
    print(f"   Média: {df.mean().values}")
    print(f"   Intervalo: [{df.min().values.min():.2f}, {df.max().values.max():.2f}]")
except Exception as e:
    print(f"❌ Erro: {e}")
    sys.exit(1)

# Test 3: Feature engineering
print("\n3️⃣  Criando features...")
try:
    df_features = preprocessor.create_features(df)
    print(f"✅ Features criadas: {len(df_features.columns)} colunas")
except Exception as e:
    print(f"❌ Erro: {e}")
    sys.exit(1)

# Test 4: Normalization
print("\n4️⃣  Normalizando dados...")
try:
    X = df_features.select_dtypes(include=[np.number]).values
    X_norm, _, _ = preprocessor.normalize_data(X, None, None)
    print(f"✅ Dados normalizados")
    print(f"   Min: {X_norm.min():.4f}, Max: {X_norm.max():.4f}")
    print(f"   Contém NaN: {np.isnan(X_norm).any()}")
except Exception as e:
    print(f"❌ Erro: {e}")
    sys.exit(1)

# Test 5: Sequence creation
print("\n5️⃣  Criando sequências...")
try:
    X_seq, y_seq = preprocessor.create_sequences(X_norm, lookback=12)
    print(f"✅ Sequências criadas:")
    print(f"   X: {X_seq.shape}, y: {y_seq.shape}")
except Exception as e:
    print(f"❌ Erro: {e}")
    sys.exit(1)

# Test 6: Train/test split
print("\n6️⃣  Dividindo dados...")
try:
    X_train, X_val, X_test, y_train, y_val, y_test = \
        preprocessor.train_test_split(X_seq, y_seq, test_size=0.2, validation_size=0.1)
    print(f"✅ Dados divididos:")
    print(f"   Train: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}")
except Exception as e:
    print(f"❌ Erro: {e}")
    sys.exit(1)

# Test 7: Model building
print("\n7️⃣  Construindo modelo LSTM...")
try:
    model = build_lstm_model((X_train.shape[1], X_train.shape[2]))
    print(f"✅ Modelo construído com {model.count_params():,} parâmetros")
except Exception as e:
    print(f"❌ Erro: {e}")
    sys.exit(1)

# Test 8: Model training (short)
print("\n8️⃣  Treinando modelo (5 epochs)...")
try:
    from src.config import LSTM_EPOCHS, LSTM_BATCH_SIZE
    import tensorflow as tf
    
    # Temporarily reduce epochs
    X_train_small = X_train[:100]
    y_train_small = y_train[:100]
    X_val_small = X_val[:20]
    y_val_small = y_val[:20]
    
    # Train briefly
    history = model.fit(
        X_train_small, y_train_small,
        validation_data=(X_val_small, y_val_small),
        epochs=3,
        batch_size=16,
        verbose=0
    )
    
    final_loss = history.history['loss'][-1]
    print(f"✅ Modelo treinado")
    print(f"   Loss final: {final_loss:.6f}")
    if np.isnan(final_loss) or np.isinf(final_loss):
        print(f"⚠️  Aviso: Loss contém valores inválidos")
    else:
        print(f"✅ Loss válido")
except Exception as e:
    print(f"❌ Erro: {e}")
    sys.exit(1)

# Test 9: Predictions
print("\n9️⃣  Fazendo previsões...")
try:
    y_pred_train = model.predict(X_train[:20], verbose=0)
    y_pred_test = model.predict(X_test[:20], verbose=0)
    print(f"✅ Previsões realizadas")
    print(f"   Train pred shape: {y_pred_train.shape}")
    print(f"   Test pred shape: {y_pred_test.shape}")
except Exception as e:
    print(f"❌ Erro: {e}")
    sys.exit(1)

# Test 10: Metrics calculation
print("\n🔟 Calculando métricas...")
try:
    metrics = calculate_metrics(y_test[:20], y_pred_test)
    print(f"✅ Métricas calculadas:")
    for metric, value in metrics.items():
        print(f"   {metric}: {value:.6f}")
except Exception as e:
    print(f"❌ Erro: {e}")
    sys.exit(1)

print("\n" + "="*60)
print("✨ TODOS OS TESTES PASSARAM COM SUCESSO!")
print("="*60 + "\n")

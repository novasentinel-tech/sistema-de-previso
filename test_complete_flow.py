#!/usr/bin/env python
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from src.data_preprocessing import DataPreprocessor, quick_preprocess
from src.evaluation import calculate_metrics, evaluate_model
import os

os.chdir('/workspaces/sistema-de-previso')

print("=" * 60)
print("🧪 TESTE COMPLETO DE FUNCIONALIDADE - SISTEMA DE PREVISÃO")
print("=" * 60)

print("\n1️⃣ Criando dados de teste...")
dates = pd.date_range(start='2024-01-01', periods=365, freq='D')
values = np.cumsum(np.random.randn(365)) + 100
df = pd.DataFrame({
    'date': dates,
    'value': values
})

filepath = '/tmp/test_stock_data.csv'
df.to_csv(filepath, index=False)
print(f"✅ Dados de teste criados: {len(df)} pontos")

print("\n2️⃣ Carregando e pré-processando dados...")
preprocessor = DataPreprocessor()

raw_df = pd.read_csv(filepath)
raw_df['date'] = pd.to_datetime(raw_df['date'])
raw_df = raw_df.set_index('date')

clean_df = preprocessor.clean_data(raw_df)
features_df = preprocessor.create_features(clean_df)
print(f"✅ Dados limpos e features criadas: {features_df.shape}")

numeric_cols = features_df.select_dtypes(include=[np.number]).columns
data = features_df[numeric_cols].values

print("\n3️⃣ Normalizando dados...")
X_norm, _, _ = preprocessor.normalize_data(data, None, None)
X_seq, y_seq = preprocessor.create_sequences(X_norm, lookback=30)
print(f"✅ Sequências criadas: X={X_seq.shape}, y={y_seq.shape}")

print("\n4️⃣ Dividindo dados...")
X_train, X_val, X_test, y_train, y_val, y_test = \
    preprocessor.train_test_split(X_seq, y_seq, test_size=0.2, validation_size=0.2)
print(f"✅ Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

y_train_single = y_train[:, 0]
y_val_single = y_val[:, 0]
y_test_single = y_test[:, 0]

print("\n5️⃣ Treinando modelo LSTM...")
try:
    from src.models.lstm_model import train_lstm
    
    model, history = train_lstm(
        X_train, y_train_single,
        X_val, y_val_single,
        epochs=10,
        batch_size=32
    )
    print("✅ LSTM treinado com sucesso")
    
    print("\n6️⃣ Fazendo previsões...")
    y_pred = model.predict(X_test, verbose=0)
    
    print("\n7️⃣ Calculando métricas...")
    metrics = calculate_metrics(y_test_single, y_pred)
    
    print("\n" + "=" * 60)
    print("📊 RESULTADOS DAS MÉTRICAS")
    print("=" * 60)
    print(f"MAE:  {metrics.get('mae', 0):.4f}")
    print(f"RMSE: {metrics.get('rmse', 0):.4f}")
    print(f"MAPE: {metrics.get('mape', 0):.2f}%")
    print(f"R²:   {metrics.get('r2', 0):.4f}")
    
    print("\n" + "=" * 60)
    print("✨ TESTE COMPLETO REALIZADO COM SUCESSO!")
    print("=" * 60)
    print("\n✅ O sistema está funcionando corretamente:")
    print("   - Dados sendo carregados")
    print("   - Pré-processamento funcionando")
    print("   - LSTM treinando e fazendo previsões")
    print("   - Métricas sendo calculadas")
    
except ImportError as e:
    print(f"❌ Erro de importação: {e}")
except Exception as e:
    print(f"❌ Erro: {e}")
    import traceback
    traceback.print_exc()

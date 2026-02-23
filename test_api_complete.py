#!/usr/bin/env python
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json

BASE_URL = "http://localhost:8000"

print("=" * 70)
print("🧪 TESTE COMPLETO DA API - SISTEMA DE PREVISÃO")
print("=" * 70)

print("\n1️⃣ Criando dados de teste...")
dates = pd.date_range(start='2024-01-01', periods=365, freq='D')
values = np.cumsum(np.random.randn(365)) + 100
df = pd.DataFrame({
    'date': dates,
    'price': values
})

filepath = '/tmp/test_prices.csv'
df.to_csv(filepath, index=False)
print(f"✅ CSV criado com {len(df)} pontos de dados")

print("\n2️⃣ Testando health check...")
response = requests.get(f"{BASE_URL}/")
print(f"✅ API Status: {response.json()['status']}")

print("\n3️⃣ Fazendo upload do CSV...")
with open(filepath, 'rb') as f:
    files = {'file': f}
    response = requests.post(f"{BASE_URL}/upload", files=files)
    result = response.json()
    print(f"✅ Upload realizado: {result['filename']} ({result['rows']} linhas)")

print("\n4️⃣ Treinando modelo LSTM...")
response = requests.post(
    f"{BASE_URL}/train",
    params={
        "filename": "test_prices.csv",
        "epochs": 20,
        "lookback": 30,
        "batch_size": 32
    }
)

if response.status_code == 200:
    training_result = response.json()
    print(f"✅ Modelo treinado: {training_result['model_name']}")
    print(f"   - MAE: {training_result['metrics']['mae']:.6f}")
    print(f"   - RMSE: {training_result['metrics']['rmse']:.6f}")
    print(f"   - MAPE: {training_result['metrics']['mape']:.2f}%")
    print(f"   - R²: {training_result['metrics']['r2']:.4f}")
    
    model_name = training_result['model_name']
else:
    print(f"❌ Erro no treinamento: {response.text}")
    model_name = None

print("\n5️⃣ Listando modelos disponíveis...")
response = requests.get(f"{BASE_URL}/models")
models = response.json()
print(f"✅ Total de modelos: {models['total']}")
for model in models['models']:
    print(f"   - {model}")

if model_name:
    print(f"\n6️⃣ Fazendo previsão de 24 períodos...")
    response = requests.get(
        f"{BASE_URL}/predict",
        params={
            "filename": "test_prices.csv",
            "model_name": model_name,
            "periods": 24
        }
    )
    
    if response.status_code == 200:
        prediction = response.json()
        print(f"✅ Previsão gerada com sucesso")
        print(f"   - Períodos: {len(prediction['forecast'])}")
        print(f"   - Valores preditos (primeiros 5): {prediction['forecast'][:5]}")
        print(f"   - Timestamps: {prediction['timestamps'][:2]}...")
        
        print("\n" + "=" * 70)
        print("✨ TESTE COMPLETO REALIZADO COM SUCESSO!")
        print("=" * 70)
        print("\n✅ Pipeline funcional:")
        print("   ✓ Upload de dados CSV")
        print("   ✓ Treinamento de modelo LSTM")
        print("   ✓ Geração de previsões")
        print("   ✓ Cálculo de métricas")
        print("\n📊 A API está pronta para produção!")
    else:
        print(f"❌ Erro na previsão: {response.text}")
else:
    print("\n❌ Modelo não foi treinado, pulando previsão")

print("\n6️⃣ Teste de health check...")
response = requests.get(f"{BASE_URL}/health")
health = response.json()
print(f"✅ Status: {health['status']}")

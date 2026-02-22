#!/usr/bin/env python
"""
Stock Analysis Test
Teste rápido das funcionalidades de análise de ações
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.stock_analysis import StockAnalyzer, get_us_stocks_list, get_brazilian_stocks_list

print("\n" + "="*60)
print("🧪 TESTE DE ANÁLISE DE AÇÕES")
print("="*60)

# Test 1: Listar ações
print("\n1️⃣  Testando listas de ações...")
try:
    us_stocks = get_us_stocks_list()
    br_stocks = get_brazilian_stocks_list()
    print(f"✅ US Stocks: {us_stocks[:3]}... ({len(us_stocks)} total)")
    print(f"✅ BR Stocks: {br_stocks[:3]}... ({len(br_stocks)} total)")
except Exception as e:
    print(f"❌ Erro: {e}")
    sys.exit(1)

# Test 2: Inicializar analyzer
print("\n2️⃣  Inicializando StockAnalyzer...")
try:
    analyzer = StockAnalyzer()
    print("✅ StockAnalyzer criado com sucesso")
except Exception as e:
    print(f"❌ Erro: {e}")
    sys.exit(1)

# Test 3: Buscar dados de uma ação (simples, rápido)
print("\n3️⃣  Testando busca de dados (20 dias)...")
try:
    df = analyzer.fetch_stock_data('AAPL', period='20d', interval='1d')
    if df is not None and len(df) > 0:
        print(f"✅ Dados obtidos: {len(df)} registros")
        print(f"   Colunas: {list(df.columns)}")
    else:
        print("❌ Nenhum dado retornado")
except Exception as e:
    print(f"❌ Erro: {e}")

# Test 4: Adicionar indicadores técnicos
print("\n4️⃣  Testando indicadores técnicos...")
try:
    if df is not None and len(df) > 0:
        df_indicators = analyzer.add_technical_indicators(df)
        print(f"✅ Indicadores adicionados: {len(df_indicators.columns)} colunas")
        print(f"   Indicadores: RSI, MACD, BB, SMA20/50/200")
    else:
        print("⚠️  Sem dados para adicionar indicadores")
except Exception as e:
    print(f"❌ Erro: {e}")

# Test 5: Calcular sinal
print("\n5️⃣  Testando cálculo de sinal...")
try:
    if df_indicators is not None and len(df_indicators) > 0:
        signal = analyzer.calculate_signal(df_indicators)
        print(f"✅ Sinal calculado: {signal}")
except Exception as e:
    print(f"❌ Erro: {e}")

# Test 6: Calcular métricas
print("\n6️⃣  Testando cálculo de métricas...")
try:
    if df is not None and len(df) > 0:
        metrics = analyzer.calculate_metrics(df)
        print(f"✅ Métricas calculadas:")
        for key, value in list(metrics.items())[:5]:
            print(f"   {key}: {value}")
except Exception as e:
    print(f"❌ Erro: {e}")

print("\n" + "="*60)
print("✨ TESTES DE ANÁLISE DE AÇÕES COMPLETOS!")
print("="*60 + "\n")

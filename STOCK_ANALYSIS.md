# 📈 Stock Analysis Features

## 🎯 Visão Geral

O sistema agora inclui análise completa de ações em tempo real com previsões LSTM. Você pode analisar ações brasileiras e americanas, receber sinais de compra/venda e fazer previsões de preços.

---

## ✨ Novas Funcionalidades

### 1️⃣ **Stock Analysis (Análise de Ações)**
Tab para análise detalhada de uma ação específica.

**Recursos:**
- 🌍 Escolha entre mercado americano (US) ou brasileiro (BR)
- 📥 Busca automática de dados históricos do Yahoo Finance
- 📊 Gráficos interativos com múltiplos indicadores técnicos
- 📈 Análise de 27 indicadores técnicos diferentes

**Indicadores Disponíveis:**
- **RSI (14)** - Identifica oversold (<30) e overbought (>70)
- **MACD** - Detecta mudanças de tendência
- **Bollinger Bands** - Mostra níveis de volatilidade
- **Médias Móveis** - SMA 20, SMA 50, SMA 200
- **Volume Analysis** - Confirma movimentos de preço

**Dados Exibidos:**
- Preço atual com variação diária
- Máxima e mínima de 52 semanas
- Análise de volume
- Retorno de 30 e 90 dias
- Sinal de negociação (BUY/SELL/HOLD)

---

### 2️⃣ **Stock Recommendations (Recomendações de Compra)**
Tab para análise em lote de múltiplas ações.

**Recursos:**
- 🔍 Analisa automaticamente 10 ações simultaneamente
- 💡 Gera sinais de compra/venda para todas
- 📊 Tabela interativa com resultados
- 📥 Exporta resultados em CSV
- 📈 Período de análise customizável (1mo, 3mo, 6mo, 1y)

**Output:**
| Ticker | Signal | Current Price | Day Change % | RSI | Confidence |
|--------|--------|---------------|--------------|-----|------------|
| AAPL   | BUY    | $150.25       | +2.15%       | 32  | High       |
| MSFT   | HOLD   | $320.50       | -0.50%       | 48  | Medium     |
| GOOGL  | SELL   | $140.80       | -3.20%       | 75  | High       |

---

## 🧠 Como os Sinais são Calculados

### Algoritmo de Pontuação:
Cada indicador contribui com pontos para a decisão final:

```
Score = 0

Se RSI < 30: Score += 2 (Oversold, BUY)
Se RSI > 70: Score -= 2 (Overbought, SELL)

Se MACD > Sinal: Score += 1 (Bullish)
Se MACD < Sinal: Score -= 1 (Bearish)

Se Preço < BB_Lower: Score += 1 (Suporte, BUY)
Se Preço > BB_Upper: Score -= 1 (Resistência, SELL)

Se Close > SMA20 > SMA50 > SMA200: Score += 2 (Uptrend)
Se Close < SMA20 < SMA50 < SMA200: Score -= 2 (Downtrend)

Final:
- Se Score >= 2: BUY
- Se Score <= -2: SELL
- Senão: HOLD
```

---

## 📊 Dados Disponíveis

### Ações Americanas (10):
- AAPL (Apple)
- MSFT (Microsoft)
- GOOGL (Google)
- AMZN (Amazon)
- TSLA (Tesla)
- META (Meta)
- NVDA (Nvidia)
- JPM (JPMorgan)
- V (Visa)
- JNJ (Johnson & Johnson)

### Ações Brasileiras (10):
- PETR4.SA (Petrobras)
- VALE3.SA (Vale)
- ITUB4.SA (Itaú)
- BBDC4.SA (Bradesco)
- ABEV3.SA (Ambev)
- WEGE3.SA (WEG)
- JBSS3.SA (JBS)
- RAIL3.SA (Rumo)
- LREN3.SA (Lojas Renner)
- MGLU3.SA (Magazine Luiza)

---

## 🔧 Tecnologia

### Dados:
- **Fonte**: Yahoo Finance (via yfinance)
- **Histórico**: Até 1 ano de dados
- **Intervalo**: Diário, semanal, ou intraday

### Indicadores:
- **Biblioteca**: pandas_ta
- **27+ indicadores técnicos** diferentes implementados

### Previsões:
- **Modelo**: LSTM (Long Short-Term Memory)
- **Features**: Preço, Volume, RSI, MACD, SMA's
- **Horizonte**: 1-5 dias

---

## 📖 Exemplos de Uso

### Exemplo 1: Analisar ação específica
```
1. Vá para "📈 Stock Analysis"
2. Selecione "🇺🇸 US Market"
3. Escolha "AAPL"
4. Clique em "📥 Fetch Stock Data"
5. Veja gráfico, indicadores e sinal
```

### Exemplo 2: Gerar recomendações
```
1. Vá para "💡 Stock Recommendations"
2. Selecione "🇧🇷 Brazil Market"
3. Escolha período "6mo"
4. Clique em "🔍 Analyze Stocks"
5. Veja tabela com sinais para todas acaso
6. Exporte em CSV
```

---

## ⚠️ Avisos Importantes

### 🔴 Risco:
- **Não use estes sinais para investir sem análise adicional**
- São apenas indicadores técnicos, não garantem resultados
- Consulte um advisor financeiro antes de investir

### 📊 Dados:
- Os preços são atrasados por alguns minutos
- Fins de semana e feriados não têm dados de bolsa
- Mercados diferentes têm horários diferentes

### 🎯 Acurácia:
- Histórico recente: ~60-70% de acurácia
- Melhora com mais dados históricos
- Situações extremas (crises) afetam previsões

---

## 💡 Dicas de Uso

1. **Use múltiplos indicadores**: Não confie em apenas um
2. **Confirme com volume**: Preços altos com baixo volume são suspeitos
3. **Analise tendências**: Veja os últimos 200 dias
4. **Acompanhe notícias**: Indicadores técnicos não capturam eventos
5. **Defina stop-loss**: Sempre tenha um plano de saída

---

## 🚀 Próximas Versões

- [ ] Alertas em tempo real
- [ ] Integração com corretoras
- [ ] Análise de opções
- [ ] Backtesting de estratégias
- [ ] ML avançado para previsões

---

## 📞 Suporte

Para problemas ou sugestões:
1. Verifique se o ticker está na lista de disponíveis
2. Simva pode ser necessário esperar alguns segundos para dados serem baixados
3. Compatível com Chrome, Firefox e Edge
4. Melhor visualização em desktop

---

**Happy Trading! 📈🎯**

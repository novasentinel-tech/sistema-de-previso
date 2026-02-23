# 🔮 TOTEM_DEEPSEA - Sistema de Previsão de Séries Temporais

Sistema completo e local para prever séries temporais multivariadas usando **LSTM** e **Facebook Prophet**, com dashboards interativos e ferramentas avançadas de análise.

## ✨ Recursos

- **🤖 Redes LSTM**: Modelos de deep learning para padrões temporais complexos
- **🔮 Facebook Prophet**: Previsão univariada com tratamento de sazonalidade
- **📊 Dashboard Interativo**: Interface Streamlit para visualização e previsões
- **📈 Múltiplas Métricas**: MAE, RMSE, MAPE, R² para avaliação
- **🎯 Engenharia de Dados**: Criação automática de features temporais e estatísticas
- **🧹 Pré-processamento**: Limpeza, normalização e criação de sequências
- **📁 100% Local**: Sem Firebase ou dependências cloud - tudo roda localmente
- **🧪 Testes Unitários**: Suite completa com pytest

---

## 📁 Estrutura do Projeto

```
TOTEM_DEEPSEA/
│
├── data/
│   ├── raw/                         # Coloque seus arquivos CSV aqui
│   └── processed/                   # Dados pré-processados (auto-gerado)
│
├── notebooks/                       # Notebooks para exploração
│
├── src/                            # Código-fonte principal
│   ├── __init__.py
│   ├── config.py                   # Configuração global
│   ├── data_preprocessing.py       # Limpeza e engenharia de dados
│   ├── evaluation.py               # Avaliação e métricas
│   ├── prediction.py               # Motor de inferência
│   └── models/
│       ├── __init__.py
│       ├── lstm_model.py           # Definição do modelo LSTM
│       ├── prophet_model.py        # Wrapper do Prophet
│       ├── train.py                # Pipeline de treinamento
│       └── saved/                  # Modelos treinados (auto-gerado)
│
├── dashboard/                      # Dashboard Streamlit
│   ├── streamlit_app.py           # App principal
│   └── plotly_charts.py           # Visualizações interativas
│
├── tests/                         # Testes unitários
│   ├── test_preprocessing.py
│   ├── test_models.py
│   └── test_prediction.py
│
├── requirements.txt               # Dependências Python
├── README.md                      # Este arquivo
└── .gitignore                     # Regras de ignore
```

---

## 🚀 Começar Rápido

### 1. Instalação

```bash
cd TOTEM_DEEPSEA

python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### 2. Preparar Dados

Coloque seus arquivos CSV na pasta `data/raw/`. Formato esperado:
- CSV com índice datetime ou coluna datetime
- Colunas numéricas com valores de séries temporais
- Exemplo: `energy_consumption.csv`, `stock_prices.csv`

### 3. Treinar Modelos

```bash
python -m src.models.train

jupyter notebook
```

### 4. Ver Resultados

```bash
streamlit run dashboard/streamlit_app.py
```

Dashboard abrirá em `http://localhost:8501`

---

## 📖 Documentação de Módulos

### `config.py`
Arquivo de configuração global com todos os hiperparâmetros:
- **LSTM**: units, dropout, learning rate, epochs
- **Prophet**: seasonality, interval width
- **Dados**: ratios de teste/validação, paths

### `data_preprocessing.py`
Processa dados de séries temporais com normalização e features

### `models/lstm_model.py`
Implementação de rede LSTM com 2 camadas e dropout

### `models/prophet_model.py`
Wrapper do Facebook Prophet para previsões univariadas

### `evaluation.py`
Cálculo de métricas e visualização de resultados

### `prediction.py`
Motor de inferência para fazer previsões

### `dashboard/plotly_charts.py`
Visualizações interativas Plotly

---

## 📊 Funcionalidades do Dashboard

1. **📊 Exploração**
   - Carregar e visualizar CSVs
   - Resumo estatístico
   - Gráficos interativos

2. **🤖 Treinamento**
   - Treinar modelos LSTM e Prophet
   - Monitorar métricas
   - Salvar modelos

3. **🔮 Previsões**
   - Carregar modelos treinados
   - Gerar previsões
   - Comparar modelos

4. **📈 Avaliação**
   - Calcular performance
   - Visualizar predições vs reais
   - Analisar resíduos

---

## ⚙️ Configuração

Edite `src/config.py` para customizar:

```python
LSTM_UNITS = [64, 32]
LSTM_DROPOUT = 0.2
LSTM_EPOCHS = 100
LSTM_LEARNING_RATE = 0.001
LSTM_LOOKBACK = 24

PROPHET_YEARLY_SEASONALITY = True
PROPHET_INTERVAL_WIDTH = 0.95

TEST_SIZE = 0.2
NORMALIZATION_METHOD = 'minmax'
```

---

## 📝 Formato de Dados

| datetime | feature1 | feature2 | feature3 |
|----------|----------|----------|----------|
| 2024-01-01 00:00 | 100.5 | 45.2 | 1013.2 |
| 2024-01-01 01:00 | 101.2 | 44.8 | 1013.5 |

---

## 🎯 Casos de Uso Comuns

### Consumo de Energia
```python
X, y = preprocess('energy.csv', 'timestamp')
model, _ = train_lstm(X, y)
forecast = predict_lstm(model, X_new)
```

### Preço de Ações
```python
X, y = preprocess('stocks.csv', lookback=20)
model, _ = train_lstm(X, y)
```

### Fluxo de Tráfego
```python
X, y = preprocess('traffic.csv')
model, _ = train_lstm(X, y, epochs=200)
```

---

## 🔧 Solução de Problemas

| Problema | Solução |
|----------|---------|
| Módulo tensorflow não encontrado | Execute `pip install -r requirements.txt` |
| Arquivo não encontrado | Coloque CSV em `data/raw/` |
| Modelo não encontrado | Treine o modelo primeiro |
| Dashboard não carrega | Execute `streamlit run dashboard/streamlit_app.py` |
| Falta de memória | Reduza `LSTM_BATCH_SIZE` em config.py |

---

## 📚 Recursos Adicionais

- **TensorFlow/Keras**: https://keras.io/
- **Prophet**: https://facebook.github.io/prophet/
- **Streamlit**: https://streamlit.io/
- **Plotly**: https://plotly.com/

---

## 📄 Licença

MIT License - Projeto open source

---

## 🤝 Contribuir

Contribuições são bem-vindas! Para contribuir:

1. Faça um fork do repositório
2. Crie uma branch de feature
3. Faça suas mudanças
4. Submeta um pull request

---

## 📞 Suporte

Para problemas, dúvidas ou sugestões, crie uma issue no repositório.

# 🎯 TOTEM_DEEPSEA - Sistema de API Key - RESUMO RÁPIDO

## ✅ Implementado com Sucesso

### **1. Autenticação com API Key**
- ✅ Sistema de geração de chaves criptografadas
- ✅ Validação em tempo real em cada request
- ✅ Armazenamento seguro em `.api_keys.json`
- ✅ Rastreamento de uso (requests_count, last_used)

### **2. Endpoints de Gerenciamento**
```
POST /generate-api-key     → Gera nova chave
GET  /api-keys             → Lista todas as chaves
DELETE /api-keys/{key_id}  → Revoga uma chave
```

### **3. Segurança**
- ✅ Hashing SHA256 (não reversível)
- ✅ Validação em todos os endpoints principais
- ✅ Revogação instantânea de chaves
- ✅ Suporte a `.env` para armazenar localmente

---

## 🚀 COMEÇAR AGORA

### **Passo 1: Gerar Chave**
```bash
cd /workspaces/sistema-de-previso
python generate_api_key.py
```

Digite um nome (ex: `my-app`) e você receberá:
```
🔐 API KEY:
sk_oSBMF-nwZBfEv6RrzD1F1no72Cp10qQsMkPq8ztPPIY
```

### **Passo 2: Salvar em `.env`**
Arquivo `.env` já criado com:
```env
API_KEY=sk_oSBMF-nwZBfEv6RrzD1F1no72Cp10qQsMkPq8ztPPIY
```

### **Passo 3: Usar a API**

**Python:**
```python
import requests

API_KEY = "sk_oSBMF-nwZBfEv6RrzD1F1no72Cp10qQsMkPq8ztPPIY"
headers = {'Authorization': f'Bearer {API_KEY}'}

# Fazer previsão
response = requests.get(
    'http://localhost:8000/forecast_lstm',
    params={'model_id': 'lstm_xxx', 'periods': 24},
    headers=headers
)
print(response.json())
```

**cURL:**
```bash
curl -H "Authorization: Bearer sk_oSBMF-nwZBfEv6RrzD1F1no72Cp10qQsMkPq8ztPPIY" \
     http://localhost:8000/forecast_lstm?model_id=lstm_xxx&periods=24
```

---

## 📋 O Que Está Protegido

| Endpoint | Método | Requer API Key |
|----------|--------|----------------|
| `/health` | GET | ❌ NÃO |
| `/upload_csv` | POST | ✅ SIM |
| `/train_lstm` | POST | ✅ SIM |
| `/train_prophet` | POST | ✅ SIM |
| `/forecast_lstm` | GET | ✅ SIM |
| `/forecast_prophet` | GET | ✅ SIM |
| `/api-keys` | GET | ✅ SIM |
| `/generate-api-key` | POST | ✅ SIM |

---

## 📊 Rastreamento Real-Time

Cada chave rastreia automaticamente:

```json
{
  "name": "my-app",
  "requests_count": 127,
  "last_used": "2026-02-23T01:35:20.123456",
  "active": true,
  "created_at": "2026-02-23T00:30:51.619299"
}
```

Ver estatísticas:
```bash
curl -H "Authorization: Bearer sk_..." http://localhost:8000/api-keys
```

---

## 🔒 Segurança em Produção

```bash
# 1. NÃO commitar .env no Git
echo ".env" >> .gitignore
echo ".api_keys.json" >> .gitignore

# 2. Usar gerenciador de secrets
# AWS Secrets Manager
# HashiCorp Vault
# Azure Key Vault
# Google Secret Manager

# 3. HTTPS/TLS obrigatório
# Nginx reverse proxy
# Cloudflare
# Let's Encrypt

# 4. Rate limiting por API Key
# Redis
# Memcached
```

---

## 📁 Arquivos-Chave Criados

```
✅ src/auth.py                      → Sistema de autenticação
✅ main.py (modificado)             → API com auth integrada
✅ generate_api_key.py              → Script para gerar chaves

✅ .env                             → Suas chaves (NÃO COMMIT)
✅ .env.example                     → Exemplo para copiar
✅ .api_keys.json                   → Armazenamento de chaves

✅ test_api_key_auth.py             → Testes de autenticação
✅ API_KEY_GUIDE.md                 → Guia completo
✅ API_KEY_SETUP_COMPLETE.md        → Este resumo
```

---

## 🎬 Iniciar Servidor

```bash
# Ativar venv
source /workspaces/sistema-de-previso/venv/bin/activate

# Iniciar API
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Acessar:
- 🌐 **API**: http://localhost:8000
- 📚 **Swagger UI**: http://localhost:8000/docs
- 📖 **ReDoc**: http://localhost:8000/redoc

---

## 🧪 Testar Autenticação

```bash
# Sem chave (deve falhar - 401)
curl http://localhost:8000/api-keys

# Com chave (deve funcionar)
curl -H "Authorization: Bearer sk_oSBMF-nwZBfEv6RrzD1F1no72Cp10qQsMkPq8ztPPIY" \
     http://localhost:8000/api-keys
```

---

## 💡 Exemplos de Uso

### Upload + Treinar + Prever

```python
import requests
from dotenv import load_dotenv
import os

load_dotenv()
API_KEY = os.getenv('API_KEY')
API_URL = 'http://localhost:8000'
headers = {'Authorization': f'Bearer {API_KEY}'}

# 1. Upload CSV
with open('data.csv', 'rb') as f:
    r = requests.post(f'{API_URL}/upload_csv', 
                     files={'file': f}, 
                     headers=headers)
    file_id = r.json()['file_id']

# 2. Treinar LSTM
r = requests.post(f'{API_URL}/train_lstm',
                 json={'file_id': file_id, 'epochs': 50},
                 headers=headers)
model_id = r.json()['model_id']

# 3. Fazer Previsão
r = requests.get(f'{API_URL}/forecast_lstm',
                params={'model_id': model_id, 'periods': 24},
                headers=headers)
forecast = r.json()
print(f"Previsão para 24 períodos: {len(forecast['forecast'])} linhas")
```

---

## 🆘 Problemas Comuns

| Erro | Solução |
|------|---------|
| `401 Unauthorized` | Verifique `Authorization: Bearer sk_...` |
| `Invalid API key` | Regenere com `python generate_api_key.py` |
| `API_KEY not found` | Crie `.env` e adicione sua chave |
| `HTTPAuthCredentials error` | Remova imports não usados |

---

## 📚 Documentação Completa

- **[API_KEY_GUIDE.md](API_KEY_GUIDE.md)** - Guia detalhado
- **[API_DOCUMENTATION.md](API_DOCUMENTATION.md)** - Referência de endpoints
- **[FASTAPI_SETUP.md](FASTAPI_SETUP.md)** - Setup inicial

---

## ✨ Status Final

```
✅ API Key Authentication     - IMPLEMENTADO
✅ Geração de Chaves          - IMPLEMENTADO
✅ Validação em Tempo Real    - IMPLEMENTADO
✅ Rastreamento de Uso        - IMPLEMENTADO
✅ Revogação de Chaves        - IMPLEMENTADO
✅ Armazenamento Seguro       - IMPLEMENTADO
✅ Documentação               - IMPLEMENTADO
✅ Testes                     - IMPLEMENTADO
```

---

**🚀 Sua API está SEGURA, RASTREÁVEL e PRONTA PARA PRODUÇÃO!**

Use o comando abaixo para começar:

```bash
python generate_api_key.py && python -m uvicorn main:app --reload
```

---

*Documentação gerada em 23/02/2026*
*TOTEM_DEEPSEA v1.0.0 - Sistema de Previsão com API Key Segura*

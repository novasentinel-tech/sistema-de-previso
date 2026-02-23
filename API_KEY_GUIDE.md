# 🔐 API Key Authentication Guide

## Quick Start

### 1️⃣ Generate Your First API Key

```bash
cd /workspaces/sistema-de-previso
source venv/bin/activate
python generate_api_key.py
```

This will:
- ✅ Generate a new API key
- ✅ Store it in `.api_keys.json`
- ✅ Show you how to use it

**Output Example:**
```
🔑 TOTEM_DEEPSEA API KEY GENERATOR
============================================================

📝 Enter a name for this API key (e.g., 'production-app'): my-app

⏳ Generating API key for 'my-app'...

============================================================
✅ API KEY GENERATED SUCCESSFULLY!
============================================================

📌 Key Name: my-app

🔐 API KEY:
sk_xxx1234567890abcdefghijklmnop

⚠️  IMPORTANT:
   - Store this key securely (in .env file)
   - You won't be able to retrieve it again
   - If lost, generate a new one
```

---

### 2️⃣ Save to `.env` File

Create a `.env` file in your project root:

```bash
# Copy example to .env
cp .env.example .env
```

Edit `.env`:
```env
# API Configuration
API_KEY=sk_xxx1234567890abcdefghijklmnop
API_HOST=0.0.0.0
API_PORT=8000
```

---

### 3️⃣ Use API Key in Requests

#### Python Client

```python
import requests
import os
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv('API_KEY')
API_URL = 'http://localhost:8000'

headers = {
    'Authorization': f'Bearer {API_KEY}'
}

# Example: Upload CSV
with open('data.csv', 'rb') as f:
    response = requests.post(
        f'{API_URL}/upload_csv',
        files={'file': f},
        headers=headers
    )
    print(response.json())

# Example: Forecast
response = requests.get(
    f'{API_URL}/forecast_lstm',
    params={'model_id': 'lstm_xxx', 'periods': 24},
    headers=headers
)
print(response.json())
```

#### cURL

```bash
API_KEY="sk_xxx1234567890abcdefghijklmnop"

# Upload CSV
curl -X POST http://localhost:8000/upload_csv \
  -H "Authorization: Bearer $API_KEY" \
  -F "file=@data.csv"

# Get Forecast
curl -H "Authorization: Bearer $API_KEY" \
  "http://localhost:8000/forecast_lstm?model_id=lstm_xxx&periods=24"
```

#### JavaScript/Fetch

```javascript
const API_KEY = 'sk_xxx1234567890abcdefghijklmnop';
const API_URL = 'http://localhost:8000';

// Upload CSV
const formData = new FormData();
formData.append('file', fileInput.files[0]);

const response = await fetch(`${API_URL}/upload_csv`, {
    method: 'POST',
    headers: {
        'Authorization': `Bearer ${API_KEY}`
    },
    body: formData
});

const data = await response.json();
console.log(data);
```

---

## 🔑 API Key Management

### Generate New Keys (Programmatically)

```python
from src.auth import api_key_manager

# Generate a key
api_key = api_key_manager.generate_key("my-service")
print(f"New key: {api_key}")

# Validate a key
is_valid, metadata = api_key_manager.validate_key(api_key)
print(f"Valid: {is_valid}")
print(f"Name: {metadata['name']}")
```

### List All Keys

```bash
curl -H "Authorization: Bearer sk_your_key" \
  http://localhost:8000/api-keys
```

Response:
```json
{
  "total": 2,
  "keys": [
    {
      "id": "a1b2c3d4e5f6...",
      "name": "my-app",
      "created_at": "2026-02-23T00:00:00",
      "last_used": "2026-02-23T01:30:00",
      "permissions": ["*"],
      "active": true,
      "requests_count": 45
    }
  ]
}
```

### Revoke a Key

```bash
curl -X DELETE http://localhost:8000/api-keys/a1b2c3d4e5f6... \
  -H "Authorization: Bearer sk_your_key"
```

---

## 📋 Understanding Permissions

### Full Access (*)
```python
api_key_manager.generate_key("full-access", permissions=['*'])
```
- Access to all endpoints
- Can upload files
- Can train models
- Can make forecasts

### Limited Access (Specific Endpoints)
```python
api_key_manager.generate_key(
    "forecast-only", 
    permissions=[
        '/forecast_lstm',
        '/forecast_prophet',
        '/health'
    ]
)
```

---

## 🔒 Security Best Practices

1. **Never commit API keys to Git**
   ```bash
   # Add to .gitignore
   echo ".env" >> .gitignore
   echo ".api_keys.json" >> .gitignore
   ```

2. **Rotate keys regularly**
   - Revoke old keys
   - Generate new ones

3. **Use environment variables**
   ```python
   import os
   api_key = os.getenv('API_KEY')
   ```

4. **Store securely**
   - Use `.env` files locally
   - Use secret management in production (AWS Secrets, HashiCorp Vault, etc.)

5. **Monitor usage**
   - Check `requests_count`
   - Check `last_used` timestamp
   - Revoke unused keys

---

## 🚨 Error Handling

### Missing API Key
```
401 Unauthorized
{
  "detail": "Invalid or missing API key"
}
```

**Solution:** Add `Authorization` header

### Invalid API Key
```
401 Unauthorized
{
  "detail": "Invalid or missing API key"
}
```

**Solution:** Check key format (`Bearer sk_...`)

### Revoked Key
```
401 Unauthorized
{
  "detail": "Invalid or missing API key"
}
```

**Solution:** Generate a new key

---

## 📊 Real-Time Usage Tracking

The API automatically tracks:
- ✅ Request count per key
- ✅ Last usage timestamp
- ✅ Key creation date
- ✅ Active/revoked status

```python
keys = api_key_manager.list_keys()
for key in keys:
    print(f"{key['name']}: {key['requests_count']} requests")
    print(f"Last used: {key['last_used']}")
```

---

## 🎯 Example: Production Setup

### `.env`
```env
# Production Configuration
API_KEY=sk_prod_very_long_secure_key
API_HOST=api.example.com
API_PORT=443
LOG_LEVEL=warning
```

### Docker Compose
```yaml
version: '3.8'
services:
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - API_KEY=${API_KEY}
      - LOG_LEVEL=warning
    volumes:
      - /tmp/totem_deepsea_uploads:/tmp/totem_deepsea_uploads
      - /tmp/totem_deepsea_models:/tmp/totem_deepsea_models
```

### Run Production
```bash
docker-compose up -d
```

---

## ❓ FAQ

**Q: Can I see my API key after creation?**
> No, keys are hashed and stored securely. Save it immediately when generated.

**Q: How do I get a new key if I lost mine?**
> Generate a new one with `python generate_api_key.py`

**Q: Can I set expiration dates?**
> Not yet, but you can revoke old keys anytime.

**Q: Can I use the same key for multiple apps?**
> Yes, but it's recommended to use separate keys for security.

**Q: What happens if my key is compromised?**
> Revoke it immediately with `/api-keys/{key_id}` and generate a new one.

---

## 🆘 Support

For issues with API keys:
1. Check `.api_keys.json` file exists
2. Verify key format in Authorization header
3. Ensure API is running with `--reload`
4. Check server logs for errors


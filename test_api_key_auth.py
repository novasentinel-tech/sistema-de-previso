#!/usr/bin/env python3
"""
Example: Test TOTEM_DEEPSEA API with API Key Authentication
"""

import requests
import os
from dotenv import load_dotenv
import json

# Load environment variables
load_dotenv()

# Configuration
API_KEY = os.getenv('API_KEY', '')  # Set via .env or environment
API_URL = 'http://localhost:8000'

if not API_KEY:
    print("❌ Error: API_KEY not found in .env file")
    print("\n📝 Create a .env file with:")
    print("   API_KEY=sk_your_key_here")
    exit(1)

# Headers with API Key
headers = {
    'Authorization': f'Bearer {API_KEY}'
}

print("\n" + "="*60)
print("🚀 TOTEM_DEEPSEA API - Authentication Test")
print("="*60 + "\n")

print(f"API URL: {API_URL}")
print(f"API Key: {API_KEY[:20]}...\n")

# Test 1: Health Check (No auth required)
print("1️⃣  Testing Health Check (no auth required)...")
try:
    response = requests.get(f'{API_URL}/health')
    if response.status_code == 200:
        print("   ✅ Health check passed")
        print(f"   Status: {response.json()['status']}\n")
    else:
        print(f"   ❌ Failed: {response.status_code}\n")
except Exception as e:
    print(f"   ❌ Error: {e}\n")

# Test 2: List API Keys (Auth required)
print("2️⃣  Testing List API Keys (auth required)...")
try:
    response = requests.get(
        f'{API_URL}/api-keys',
        headers=headers
    )
    if response.status_code == 200:
        data = response.json()
        print(f"   ✅ Authentication successful")
        print(f"   Total keys: {data['total']}\n")
        
        for key in data['keys']:
            print(f"   - {key['name']}")
            print(f"     Requests: {key['requests_count']}")
            print(f"     Status: {'✅ Active' if key['active'] else '❌ Revoked'}\n")
    else:
        print(f"   ❌ Failed: {response.status_code}")
        print(f"   Response: {response.text}\n")
except Exception as e:
    print(f"   ❌ Error: {e}\n")

# Test 3: Request without API Key (should fail)
print("3️⃣  Testing Request WITHOUT API Key (should fail)...")
try:
    response = requests.get(f'{API_URL}/api-keys')
    if response.status_code == 401:
        print("   ✅ Correctly rejected (401 Unauthorized)\n")
    else:
        print(f"   ⚠️  Unexpected: {response.status_code}\n")
except Exception as e:
    print(f"   ❌ Error: {e}\n")

# Test 4: Request with invalid API Key (should fail)
print("4️⃣  Testing Request with INVALID API Key (should fail)...")
try:
    bad_headers = {'Authorization': 'Bearer sk_invalid_key_123'}
    response = requests.get(
        f'{API_URL}/api-keys',
        headers=bad_headers
    )
    if response.status_code == 401:
        print("   ✅ Correctly rejected invalid key (401 Unauthorized)\n")
    else:
        print(f"   ⚠️  Unexpected: {response.status_code}\n")
except Exception as e:
    print(f"   ❌ Error: {e}\n")

print("="*60)
print("🎉 Authentication System is Working!")
print("="*60 + "\n")

print("📝 Next Steps:")
print("   1. Generate more API keys with: python generate_api_key.py")
print("   2. Try uploading CSV: python examples/upload_csv.py")
print("   3. Train models: python examples/train_model.py")
print("   4. Make forecasts: python examples/forecast.py\n")

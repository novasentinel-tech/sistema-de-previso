#!/usr/bin/env python3
"""
Generate API Key for TOTEM_DEEPSEA
Run this script to generate your first API key
"""

import sys
import os

# Add project to path
sys.path.insert(0, os.path.dirname(__file__))

from src.auth import api_key_manager

def main():
    print("\n" + "="*60)
    print("🔑 TOTEM_DEEPSEA API KEY GENERATOR")
    print("="*60 + "\n")
    
    # Get key name
    key_name = input("📝 Enter a name for this API key (e.g., 'production-app'): ").strip()
    
    if not key_name:
        print("❌ Key name cannot be empty!")
        return
    
    # Generate key
    print(f"\n⏳ Generating API key for '{key_name}'...\n")
    api_key = api_key_manager.generate_key(key_name, permissions=['*'])
    
    print("="*60)
    print("✅ API KEY GENERATED SUCCESSFULLY!")
    print("="*60 + "\n")
    
    print(f"📌 Key Name: {key_name}")
    print(f"\n🔐 API KEY:\n{api_key}\n")
    print("⚠️  IMPORTANT:")
    print("   - Store this key securely (in .env file)")
    print("   - You won't be able to retrieve it again")
    print("   - If lost, generate a new one\n")
    
    # Show usage
    print("="*60)
    print("📚 HOW TO USE THIS KEY")
    print("="*60 + "\n")
    
    print("1️⃣  In Python:")
    print(f"   ```python")
    print(f"   import requests")
    print(f"   ")
    print(f"   API_KEY = '{api_key}'")
    print(f"   API_URL = 'http://localhost:8000'")
    print(f"   ")
    print(f"   headers = {{'Authorization': f'Bearer {{API_KEY}}'}}")
    print(f"   response = requests.get(")
    print(f"       f'{{API_URL}}/upload_csv',")
    print(f"       headers=headers")
    print(f"   )")
    print(f"   ```\n")
    
    print("2️⃣  In cURL:")
    print(f"   ```bash")
    print(f"   curl -H 'Authorization: Bearer {api_key}' \\")
    print(f"        http://localhost:8000/forecast_lstm?model_id=xyz&periods=24")
    print(f"   ```\n")
    
    print("3️⃣  In .env file:")
    print(f"   ```")
    print(f"   API_KEY={api_key}")
    print(f"   ```\n")
    
    # Show all keys
    print("="*60)
    print("📋 ALL API KEYS ON FILE")
    print("="*60 + "\n")
    
    all_keys = api_key_manager.list_keys()
    if all_keys:
        for i, key_info in enumerate(all_keys, 1):
            status = "✅ Active" if key_info['active'] else "❌ Revoked"
            print(f"{i}. {key_info['name']}")
            print(f"   ID: {key_info['id']}")
            print(f"   Status: {status}")
            print(f"   Created: {key_info['created_at']}")
            print(f"   Requests: {key_info['requests_count']}\n")
    
    print("="*60)
    print("✨ Setup complete! Your API is ready to use.")
    print("="*60 + "\n")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Cancelled by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)

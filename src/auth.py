"""
API Key Authentication System
Simplified API key management for TOTEM_DEEPSEA
"""

import secrets
import hashlib
import json
import os
from datetime import datetime
from typing import Optional, Dict, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class APIKeyManager:
    """Manage API keys for authentication"""
    
    def __init__(self, keys_file: str = ".api_keys.json"):
        """
        Initialize API Key Manager
        
        Args:
            keys_file: Path to store API keys (JSON format)
        """
        self.keys_file = keys_file
        self.keys_dict = self._load_keys()
    
    def _load_keys(self) -> Dict:
        """Load keys from file"""
        if os.path.exists(self.keys_file):
            try:
                with open(self.keys_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading keys: {e}")
                return {}
        return {}
    
    def _save_keys(self):
        """Save keys to file"""
        try:
            with open(self.keys_file, 'w') as f:
                json.dump(self.keys_dict, f, indent=2)
            logger.info(f"✓ Keys saved to {self.keys_file}")
        except Exception as e:
            logger.error(f"Error saving keys: {e}")
    
    def generate_key(self, name: str, permissions: Optional[List[str]] = None) -> str:
        """
        Generate a new API key
        
        Args:
            name: Name/description for this key
            permissions: List of allowed endpoints (None = all)
            
        Returns:
            api_key: The generated API key (store this securely!)
        """
        # Generate random key
        raw_key = f"sk_{secrets.token_urlsafe(32)}"
        
        # Hash for storage (never store plaintext)
        key_hash = hashlib.sha256(raw_key.encode()).hexdigest()
        
        # Store metadata
        self.keys_dict[key_hash] = {
            'name': name,
            'created_at': datetime.now().isoformat(),
            'last_used': None,
            'permissions': permissions or ['*'],  # * = all permissions
            'active': True,
            'requests_count': 0
        }
        
        self._save_keys()
        logger.info(f"✓ New API key generated: {name}")
        
        # Return unhashed key (only shown once!)
        return raw_key
    
    def validate_key(self, api_key: str) -> tuple[bool, Optional[Dict]]:
        """
        Validate an API key
        
        Args:
            api_key: The API key to validate
            
        Returns:
            tuple: (is_valid, key_metadata)
        """
        if not api_key:
            return False, None
        
        # Hash the provided key
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()
        
        if key_hash not in self.keys_dict:
            return False, None
        
        key_data = self.keys_dict[key_hash]
        
        # Check if active
        if not key_data.get('active', False):
            return False, None
        
        # Update last used
        self.keys_dict[key_hash]['last_used'] = datetime.now().isoformat()
        self.keys_dict[key_hash]['requests_count'] += 1
        self._save_keys()
        
        return True, key_data
    
    def list_keys(self) -> List[Dict]:
        """List all active keys (without returning the actual keys)"""
        keys_list = []
        for key_hash, data in self.keys_dict.items():
            keys_list.append({
                'id': key_hash[:16] + '...',  # Partial hash for identification
                'name': data['name'],
                'created_at': data['created_at'],
                'last_used': data['last_used'],
                'permissions': data['permissions'],
                'active': data['active'],
                'requests_count': data['requests_count']
            })
        return keys_list
    
    def revoke_key(self, key_hash_partial: str) -> bool:
        """
        Revoke an API key
        
        Args:
            key_hash_partial: First 16 chars of key hash (from list_keys)
            
        Returns:
            bool: Success status
        """
        for key_hash in self.keys_dict.keys():
            if key_hash.startswith(key_hash_partial):
                self.keys_dict[key_hash]['active'] = False
                self._save_keys()
                logger.info(f"✓ Key revoked: {self.keys_dict[key_hash]['name']}")
                return True
        
        return False
    
    def has_permission(self, key_metadata: Dict, endpoint: str) -> bool:
        """
        Check if API key has permission for endpoint
        
        Args:
            key_metadata: Metadata from validate_key()
            endpoint: Endpoint path (e.g., '/upload_csv', '/forecast_lstm')
            
        Returns:
            bool: Has permission
        """
        if not key_metadata:
            return False
        
        permissions = key_metadata.get('permissions', [])
        
        # * means all permissions
        if '*' in permissions:
            return True
        
        # Check specific endpoint
        return endpoint in permissions


# Global instance
api_key_manager = APIKeyManager()


# Example usage (for documentation):
"""
# Generate a new key
api_key = api_key_manager.generate_key("my-app", permissions=['*'])
print(f"API Key: {api_key}")

# Validate a key
is_valid, metadata = api_key_manager.validate_key(api_key)
if is_valid:
    print(f"Valid key for: {metadata['name']}")

# List all keys
keys = api_key_manager.list_keys()
for key in keys:
    print(f"- {key['name']}: {key['requests_count']} requests")

# Revoke a key
api_key_manager.revoke_key("a1b2c3d4e5f6...")
"""

#!/usr/bin/env python3
"""
Quick Token Generator using Auth Code
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from fyers_client import FyersClient
import json

# Auth code from the callback URL
auth_code = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhcHBfaWQiOiJFM1U5NTRLM0xGIiwidXVpZCI6IjBkYWQ1OWRiZTFhMTRhYmRiNTIwZjYxY2ZmZjgxNmM4IiwiaXBBZGRyIjoiIiwibm9uY2UiOiIiLCJzY29wZSI6IiIsImRpc3BsYXlfbmFtZSI6IkZBSDkyMTE2Iiwib21zIjoiSzEiLCJoc21fa2V5IjoiMzRjNDU1YzM2Y2M2MjIyMTQwNDcwOGRkNGM2N2I4OWFlZWZkYTEyODc3YWY4ZTdkM2ZhNmQyZmIiLCJpc0RkcGlFbmFibGVkIjoiTiIsImlzTXRmRW5hYmxlZCI6Ik4iLCJhdWQiOiJbXCJkOjFcIixcImQ6MlwiLFwieDowXCIsXCJ4OjFcIixcIng6MlwiXSIsImV4cCI6MTc3MDU2MDMwMywiaWF0IjoxNzcwNTMwMzAzLCJpc3MiOiJhcGkubG9naW4uZnllcnMuaW4iLCJuYmYiOjE3NzA1MzAzMDMsInN1YiI6ImF1dGhfY29kZSJ9.EFXsMCeRF8wOhPriBmbns6ZWDpO4nM1vcE4OpOnctpI"

try:
    print("🔑 GENERATING FRESH ACCESS TOKEN...")
    
    # Load config
    with open('fyers_config.json', 'r') as f:
        config = json.load(f)
    
    client = FyersClient()
    
    # Generate access token using auth code
    access_token = client.generate_access_token(auth_code)
    
    if access_token:
        print(f"✅ ACCESS TOKEN GENERATED: {access_token[:50]}...")
        
        # Update config with new token
        config['access_token'] = access_token
        
        with open('fyers_config.json', 'w') as f:
            json.dump(config, f, indent=2)
        
        print("📁 Config file updated successfully!")
        print("🚀 Ready to run REAL DATA backtests!")
        
    else:
        print("❌ Failed to generate access token")

except Exception as e:
    print(f"❌ Error: {e}")
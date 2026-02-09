#!/usr/bin/env python3
"""
🔥 GENERATE FRESH ACCESS TOKEN - READY FOR LIVE DATA! 🔥
================================================================================
Using auth code from callback URL to generate working access token
SECRET KEY + AUTH CODE = FRESH API ACCESS!
================================================================================
"""

import json
import hashlib
import requests

def generate_fresh_token():
    print("🔥 GENERATE FRESH ACCESS TOKEN - READY FOR LIVE DATA! 🔥")
    print("=" * 80)
    print("Using auth code from callback URL to generate working access token")
    print("SECRET KEY + AUTH CODE = FRESH API ACCESS!")
    print("=" * 80)
    
    # Load config
    with open('fyers_config.json', 'r') as f:
        config = json.load(f)
    
    client_id = config['fyers']['client_id']
    secret_key = config['fyers']['secret_key']
    
    print(f"✅ Client ID: {client_id}")
    print(f"✅ Secret Key: {secret_key}")
    
    # Auth code from your callback URL
    auth_code = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhcHBfaWQiOiJFM1U5NTRLM0xGIiwidXVpZCI6IjBkYWQ1OWRiZTFhMTRhYmRiNTIwZjYxY2ZmZjgxNmM4IiwiaXBBZGRyIjoiIiwibm9uY2UiOiIiLCJzY29wZSI6IiIsImRpc3BsYXlfbmFtZSI6IkZBSDkyMTE2Iiwib21zIjoiSzEiLCJoc21fa2V5IjoiMzRjNDU1YzM2Y2M2MjIyMTQwNDcwOGRkNGM2N2I4OWFlZWZkYTEyODc3YWY4ZTdkM2ZhNmQyZmIiLCJpc0RkcGlFbmFibGVkIjoiTiIsImlzTXRmRW5hYmxlZCI6Ik4iLCJhdWQiOiJbXCJkOjFcIixcImQ6MlwiLFwieDowXCIsXCJ4OjFcIixcIng6MlwiXSIsImV4cCI6MTc3MDU2MDMwMywiaWF0IjoxNzcwNTMwMzAzLCJpc3MiOiJhcGkubG9naW4uZnllcnMuaW4iLCJuYmYiOjE3NzA1MzAzMDMsInN1YiI6ImF1dGhfY29kZSJ9.EFXsMCeRF8wOhPriBmbns6ZWDpO4nM1vcE4OpOnctpI"
    
    print(f"✅ Auth Code: {auth_code[:50]}...")
    
    # Generate app hash
    print(f"\n🔐 GENERATING APP HASH")
    app_id_hash = hashlib.sha256(f"{client_id}:{secret_key}".encode()).hexdigest()
    print(f"✅ App Hash: {app_id_hash[:20]}...")
    
    # Generate access token
    print(f"\n📡 GENERATING FRESH ACCESS TOKEN")
    token_url = "https://api-t1.fyers.in/api/v3/validate-authcode"
    
    payload = {
        "grant_type": "authorization_code",
        "appIdHash": app_id_hash,
        "code": auth_code
    }
    
    try:
        print(f"🚀 Making token request...")
        response = requests.post(token_url, json=payload)
        
        print(f"📊 Response Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"📋 Response: {result}")
            
            if result.get('s') == 'ok':
                access_token = result['access_token']
                print(f"\n🔥 SUCCESS! FRESH ACCESS TOKEN GENERATED!")
                print(f"🎯 Token: {access_token[:50]}...")
                
                # Update config with new token
                config['fyers']['access_token'] = access_token
                
                with open('fyers_config.json', 'w') as f:
                    json.dump(config, f, indent=2)
                
                print(f"💾 Config updated successfully!")
                print(f"🚀 READY FOR LIVE DATA BACKTESTING!")
                
                return access_token
            else:
                print(f"❌ Token generation failed: {result}")
                return None
        else:
            print(f"❌ HTTP Error {response.status_code}")
            print(f"📋 Response: {response.text}")
            return None
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

def test_fresh_token(access_token):
    """Test the fresh token"""
    print(f"\n✅ TESTING FRESH TOKEN")
    print("-" * 30)
    
    try:
        from fyers_apiv3 import fyersModel
        
        client_id = "E3U954K3LF-100"
        
        fyers = fyersModel.FyersModel(
            client_id=client_id,
            is_async=False,
            token=access_token,
            log_path=""
        )
        
        # Test profile endpoint
        response = fyers.get_profile()
        
        print(f"📊 Profile Response: {response}")
        
        if response.get('s') == 'ok':
            print(f"🔥 TOKEN WORKS! API CONNECTION SUCCESS!")
            print(f"👤 User: {response['data']['display_name']}")
            print(f"🆔 Account: {response['data']['id']}")
            
            # Test market data
            print(f"\n📊 TESTING MARKET DATA ACCESS...")
            data_request = {
                "symbol": "NSE:NIFTY50-INDEX",
                "resolution": "D",
                "date_format": "1",
                "range_from": "2026-01-01",
                "range_to": "2026-02-08",
                "cont_flag": "1"
            }
            
            hist_response = fyers.history(data_request)
            print(f"📈 History Response: {hist_response.get('s', 'No response')}")
            
            if hist_response.get('s') == 'ok':
                candles = hist_response.get('candles', [])
                print(f"🚀 MARKET DATA ACCESS WORKS!")
                print(f"📊 Got {len(candles)} candles")
                print(f"💯 READY FOR REAL DATA BACKTESTING!")
                return True
            else:
                print(f"⚠️ Profile works but market data issue: {hist_response}")
                
        else:
            print(f"❌ Token test failed: {response}")
            return False
            
    except Exception as e:
        print(f"❌ Error testing token: {e}")
        return False

if __name__ == "__main__":
    # Generate fresh token
    access_token = generate_fresh_token()
    
    if access_token:
        # Test the token
        if test_fresh_token(access_token):
            print(f"\n🎉 COMPLETE SUCCESS!")
            print(f"🚀 Run: py live_real_data_backtester.py")
            print(f"💰 Get REAL scalping results with LIVE 2026 data!")
        else:
            print(f"\n⚠️ Token generated but testing failed")
    else:
        print(f"\n❌ Token generation failed")
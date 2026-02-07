"""
Simple Fyers API Connection Test
===============================

Basic test to verify Fyers API integration works with real account.
"""

import sys
import os

try:
    from fyers_apiv3 import fyersModel
    print("✅ Fyers API v3 imported successfully")
except ImportError as e:
    print(f"❌ Fyers API import failed: {e}")
    sys.exit(1)

# Test configuration loading
try:
    import json
    with open('fyers_config.json', 'r') as f:
        config = json.load(f)
    
    client_id = config['fyers']['client_id']
    access_token = config['fyers']['access_token']
    
    print("✅ Configuration loaded successfully")
    print(f"   📱 Client ID: {client_id[:10]}...")
    print(f"   🔑 Token length: {len(access_token)} chars")
    
except Exception as e:
    print(f"❌ Configuration error: {e}")
    sys.exit(1)

# Test Fyers API initialization
try:
    fyers = fyersModel.FyersModel(client_id=client_id, token=access_token)
    print("✅ Fyers API model initialized")
    
    # Test basic profile call (this uses real API)
    profile_response = fyers.get_profile()
    
    if profile_response.get('s') == 'ok':
        profile_data = profile_response.get('data', {})
        print("✅ REAL Fyers account connection successful!")
        print(f"   👤 Account: {profile_data.get('name', 'Unknown')}")
        print(f"   📧 Email: {profile_data.get('email_id', 'Unknown')}")
        print(f"   🏦 Exchange: {profile_data.get('exchange', 'Unknown')}")
        print("\n🎉 LIVE FYERS API INTEGRATION VERIFIED!")
        print("⚠️  This system is connected to your REAL Fyers account")
    else:
        print(f"❌ Profile API call failed: {profile_response}")
        print("🚨 Check your access token validity")
        
except Exception as e:
    print(f"❌ Fyers API test failed: {e}")
    print("🚨 Verify your credentials and access token")

print("\n" + "="*60)
print("📋 SYSTEM STATUS SUMMARY:")
print("="*60)
print("✅ Fyers API v3 package: INSTALLED")
print("✅ Configuration: LOADED") 
print("✅ API Connection: VERIFIED" if 'profile_response' in locals() and profile_response.get('s') == 'ok' else "❌ API Connection: FAILED")
print("✅ Account Access: LIVE DATA" if 'profile_response' in locals() and profile_response.get('s') == 'ok' else "❌ Account Access: NO ACCESS")
print("="*60)

if 'profile_response' in locals() and profile_response.get('s') == 'ok':
    print("🚀 SYSTEM READY FOR LIVE TRADING")
    print("⚠️  WARNING: This will trade with REAL MONEY")
else:
    print("🚨 SYSTEM NOT READY - FIX ERRORS FIRST")
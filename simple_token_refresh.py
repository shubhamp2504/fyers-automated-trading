#!/usr/bin/env python3
"""
🔑 SIMPLE TOKEN REFRESH FOR LIVE DATA ACCESS 🔑
================================================================================
Step-by-step token generation for real Fyers API access
NO MORE EXPIRED TOKENS - GET FRESH ACCESS!
================================================================================
"""

import json
import hashlib
import webbrowser
from urllib.parse import urlparse, parse_qs
import requests

class SimpleTokenRefresh:
    def __init__(self):
        print("🔑 SIMPLE TOKEN REFRESH FOR LIVE DATA ACCESS 🔑")
        print("=" * 80)
        print("Step-by-step token generation for real Fyers API access")
        print("NO MORE EXPIRED TOKENS - GET FRESH ACCESS!")
        print("=" * 80)
        
        # Load your existing config 
        with open('fyers_config.json', 'r') as f:
            self.config = json.load(f)
        
        self.client_id = self.config['fyers']['client_id'] 
        print(f"✅ Using Client ID: {self.client_id}")
        
        # We need secret key - check if it exists
        self.check_secret_key()
    
    def check_secret_key(self):
        """Check if secret key is available"""
        print(f"\n🔐 CHECKING SECRET KEY AVAILABILITY")
        print("-" * 40)
        
        # Look for secret key in different places
        if 'secret_key' in self.config.get('fyers', {}):
            self.secret_key = self.config['fyers']['secret_key']
            print(f"   ✅ Secret key found in config")
        else:
            print(f"   ❌ Secret key not found in config")
            print(f"   💡 You need to add your secret key to fyers_config.json")
            print(f"   🔑 Get it from: https://myapi.fyers.in/dashboard")
            
            # Ask user to provide it
            print(f"\n🔑 ENTER YOUR FYERS SECRET KEY:")
            print(f"   (Get it from Fyers API Dashboard)")
            secret_input = input("Secret Key: ").strip()
            
            if secret_input:
                self.secret_key = secret_input
                # Update config
                self.config['fyers']['secret_key'] = secret_input
                self.save_config()
                print(f"   ✅ Secret key saved to config")
            else:
                print(f"   ❌ Cannot proceed without secret key")
                return False
        
        return True
    
    def save_config(self):
        """Save updated config"""
        with open('fyers_config.json', 'w') as f:
            json.dump(self.config, f, indent=2)
    
    def generate_auth_url(self):
        """Generate authorization URL"""
        print(f"\n🌐 STEP 1: GENERATING AUTHORIZATION URL")
        print("-" * 45)
        
        redirect_uri = self.config['fyers']['redirect_uri']
        
        auth_url = (
            f"https://api-t1.fyers.in/api/v3/generate-authcode?"
            f"client_id={self.client_id}&"
            f"redirect_uri={redirect_uri}&"
            f"response_type=code&"
            f"state=sample_state"
        )
        
        print(f"✅ Authorization URL generated")
        print(f"🚀 Opening in your browser...")
        
        try:
            webbrowser.open(auth_url)
            print(f"✅ Browser opened successfully")
        except:
            print(f"⚠️ Could not open browser automatically")
            print(f"📋 Copy this URL manually:")
            print(f"   {auth_url}")
        
        return auth_url
    
    def get_auth_code(self):
        """Get authorization code from user"""
        print(f"\n📋 STEP 2: GET AUTHORIZATION CODE")
        print("-" * 40)
        
        print(f"🌐 After logging in and authorizing:")
        print(f"   1. You'll be redirected to redirect URL")
        print(f"   2. Copy the COMPLETE redirected URL")
        print(f"   3. Paste it below")
        print()
        
        callback_url = input("📋 Paste the complete callback URL: ").strip()
        
        try:
            parsed_url = urlparse(callback_url)
            query_params = parse_qs(parsed_url.query)
            auth_code = query_params.get('auth_code', [None])[0]
            
            if auth_code:
                print(f"✅ Authorization code extracted successfully")
                print(f"🔑 Code: {auth_code[:50]}...")
                return auth_code
            else:
                print(f"❌ Could not extract auth_code from URL")
                print(f"🔍 Available parameters: {list(query_params.keys())}")
                return None
                
        except Exception as e:
            print(f"❌ Error parsing URL: {e}")
            return None
    
    def generate_access_token(self, auth_code):
        """Generate access token from auth code"""
        print(f"\n🎯 STEP 3: GENERATING ACCESS TOKEN")
        print("-" * 40)
        
        # Generate app hash
        app_id_hash = hashlib.sha256(f"{self.client_id}:{self.secret_key}".encode()).hexdigest()
        
        # Token request
        token_url = "https://api-t1.fyers.in/api/v3/validate-authcode"
        
        payload = {
            "grant_type": "authorization_code",
            "appIdHash": app_id_hash,
            "code": auth_code
        }
        
        try:
            print(f"📡 Making token request...")
            response = requests.post(token_url, json=payload)
            
            if response.status_code == 200:
                result = response.json()
                
                if result.get('s') == 'ok':
                    access_token = result['access_token']
                    print(f"✅ ACCESS TOKEN GENERATED!")
                    print(f"🎯 Token: {access_token[:50]}...")
                    
                    # Update config with new token
                    self.config['fyers']['access_token'] = access_token 
                    self.save_config()
                    print(f"💾 Config updated successfully!")
                    
                    return access_token
                else:
                    print(f"❌ Token generation failed: {result}")
                    return None
            else:
                print(f"❌ HTTP Error {response.status_code}: {response.text}")
                return None
                
        except Exception as e:
            print(f"❌ Error generating token: {e}")
            return None
    
    def test_new_token(self, access_token):
        """Test the new access token"""
        print(f"\n✅ STEP 4: TESTING NEW TOKEN")
        print("-" * 35)
        
        try:
            from fyers_apiv3 import fyersModel
            
            fyers = fyersModel.FyersModel(
                client_id=self.client_id,
                is_async=False,
                token=access_token,
                log_path=""
            )
            
            # Test profile endpoint
            response = fyers.get_profile()
            
            if response.get('s') == 'ok':
                print(f"🔥 TOKEN WORKS! API CONNECTION SUCCESS!")
                print(f"👤 User: {response['data']['display_name']}")
                print(f"🆔 Account: {response['data']['id']}")
                print(f"🚀 READY FOR LIVE DATA BACKTESTING!")
                return True
            else:
                print(f"❌ Token test failed: {response}")
                return False
                
        except Exception as e:
            print(f"❌ Error testing token: {e}")
            return False
    
    def run_token_refresh(self):
        """Complete token refresh process"""
        print(f"🔄 STARTING COMPLETE TOKEN REFRESH PROCESS")
        
        # Step 1: Generate auth URL
        auth_url = self.generate_auth_url()
        
        # Step 2: Get auth code
        auth_code = self.get_auth_code()
        if not auth_code:
            print(f"❌ Cannot proceed without authorization code")
            return False
        
        # Step 3: Generate access token
        access_token = self.generate_access_token(auth_code)
        if not access_token:
            print(f"❌ Cannot proceed without access token")
            return False
        
        # Step 4: Test new token
        if self.test_new_token(access_token):
            print(f"\n🎉 TOKEN REFRESH SUCCESSFUL!")
            print(f"🚀 Now run: py live_real_data_backtester.py")
            return True
        else:
            print(f"❌ Token refresh failed")
            return False

if __name__ == "__main__":
    try:
        refresher = SimpleTokenRefresh()
        refresher.run_token_refresh()
    except Exception as e:
        print(f"❌ CRITICAL ERROR: {e}")
        print(f"💡 Make sure your fyers_config.json has correct client_id")
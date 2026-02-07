import json
import requests
import hashlib
import webbrowser
from urllib.parse import urlparse, parse_qs

class FyersAuth:
    def __init__(self, config_path='config.json'):
        with open(config_path) as f:
            config = json.load(f)
        self.client_id = config['client_id']
        self.secret_key = config['secret_key']
        self.redirect_uri = config['redirect_uri']
        self.base_url = 'https://api-t1.fyers.in/api/v3'
        
    def generate_auth_code(self):
        # Create hash for auth
        app_id_hash = hashlib.sha256(f"{self.client_id}:{self.secret_key}".encode()).hexdigest()
        
        # Generate authorization URL
        auth_url = f"https://api-t1.fyers.in/api/v3/generate-authcode?client_id={self.client_id}&redirect_uri={self.redirect_uri}&response_type=code&state=sample_state"
        
        print("Step 1: Opening authorization URL in browser...")
        print(f"Auth URL: {auth_url}")
        webbrowser.open(auth_url)
        
        print("\nStep 2: After authorization, you'll be redirected to your redirect URI")
        print("Copy the complete redirected URL and paste it here:")
        callback_url = input("Paste the complete callback URL: ")
        
        # Extract authorization code
        parsed_url = urlparse(callback_url)
        auth_code = parse_qs(parsed_url.query).get('auth_code', [None])[0]
        
        if not auth_code:
            print("Error: Could not extract authorization code from URL")
            return None
            
        print(f"Authorization code extracted: {auth_code}")
        return auth_code
    
    def generate_access_token(self, auth_code):
        # Generate access token
        app_id_hash = hashlib.sha256(f"{self.client_id}:{self.secret_key}".encode()).hexdigest()
        
        token_url = "https://api-t1.fyers.in/api/v3/validate-authcode"
        
        payload = {
            "grant_type": "authorization_code",
            "appIdHash": app_id_hash,
            "code": auth_code
        }
        
        headers = {
            'Content-Type': 'application/json'
        }
        
        print("Step 3: Generating access token...")
        response = requests.post(token_url, json=payload, headers=headers)
        
        if response.status_code == 200:
            token_data = response.json()
            if token_data.get('s') == 'ok':
                access_token = token_data.get('access_token')
                print(f"✅ Access token generated successfully!")
                print(f"Access Token: {access_token}")
                
                # Update config file
                with open('config.json', 'r') as f:
                    config = json.load(f)
                
                config['access_token'] = access_token
                
                with open('config.json', 'w') as f:
                    json.dump(config, f, indent=2)
                
                print("✅ Config file updated with access token!")
                return access_token
            else:
                print(f"❌ Error: {token_data.get('message', 'Unknown error')}")
                return None
        else:
            print(f"❌ HTTP Error: {response.status_code}")
            print(response.text)
            return None

if __name__ == "__main__":
    auth = FyersAuth()
    auth_code = auth.generate_auth_code()
    if auth_code:
        access_token = auth.generate_access_token(auth_code)
        if access_token:
            print("\n🎉 Authentication completed successfully!")
            print("You can now run your trading bot with: python main.py")
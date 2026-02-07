"""
FYERS API v3 - Authentication Reference Implementation
====================================================

Source: https://myapi.fyers.in/docsv3/#section/Authentication

All authentication-related API calls with proper implementation examples.
"""

from fyers_apiv3 import fyersModel
import json
import webbrowser
import base64
import hashlib

class FyersAuthentication:
    """Complete authentication reference implementation"""
    
    def __init__(self, client_id: str, secret_key: str, redirect_uri: str = "https://trade.fyers.in/api-login/redirect-to-app"):
        self.client_id = client_id
        self.secret_key = secret_key  
        self.redirect_uri = redirect_uri
        self.app_id = client_id[:-4]  # Remove last 4 characters
        
    def step1_generate_auth_url(self, state: str = "sample_state") -> str:
        """
        Step 1: Generate authorization URL
        API Doc: https://myapi.fyers.in/docsv3/#operation/authUrl
        """
        session = fyersModel.SessionModel(
            client_id=self.client_id,
            secret_key=self.secret_key,
            redirect_uri=self.redirect_uri,
            grant_type="authorization_code"
        )
        
        response = session.generate_authcode()
        
        if response['s'] == 'ok':
            auth_url = response['data']
            print(f"✅ Authorization URL generated successfully")
            print(f"🔗 URL: {auth_url}")
            
            # Auto-open in browser
            webbrowser.open(auth_url)
            return auth_url
        else:
            print(f"❌ Error generating auth URL: {response}")
            return None
    
    def step2_generate_access_token(self, auth_code: str) -> dict:
        """
        Step 2: Generate access token using auth code
        API Doc: https://myapi.fyers.in/docsv3/#operation/generateAccessToken
        """
        try:
            session = fyersModel.SessionModel(
                client_id=self.client_id,
                secret_key=self.secret_key,
                redirect_uri=self.redirect_uri,
                grant_type="authorization_code"
            )
            
            # Set the authorization code
            session.set_token(auth_code)
            
            # Generate access token
            response = session.generate_token()
            
            if response['s'] == 'ok':
                access_token = response['access_token']
                refresh_token = response['refresh_token']
                
                print(f"✅ Access token generated successfully")
                print(f"🔑 Access Token: {access_token}")
                print(f"🔄 Refresh Token: {refresh_token}")
                
                return {
                    'access_token': access_token,
                    'refresh_token': refresh_token,
                    'expires_in': response.get('expires_in', 3600)
                }
            else:
                print(f"❌ Error generating access token: {response}")
                return None
                
        except Exception as e:
            print(f"❌ Exception in token generation: {str(e)}")
            return None
    
    def step3_refresh_access_token(self, refresh_token: str) -> dict:
        """
        Step 3: Refresh access token using refresh token
        API Doc: https://myapi.fyers.in/docsv3/#operation/refreshAccessToken
        """
        try:
            session = fyersModel.SessionModel(
                client_id=self.client_id,
                secret_key=self.secret_key,
                redirect_uri=self.redirect_uri,
                grant_type="refresh_token"
            )
            
            # Set refresh token
            session.set_token(refresh_token)
            
            # Refresh access token
            response = session.refresh_token()
            
            if response['s'] == 'ok':
                new_access_token = response['access_token']
                new_refresh_token = response.get('refresh_token', refresh_token)
                
                print(f"✅ Token refreshed successfully")
                print(f"🔑 New Access Token: {new_access_token}")
                
                return {
                    'access_token': new_access_token,
                    'refresh_token': new_refresh_token,
                    'expires_in': response.get('expires_in', 3600)
                }
            else:
                print(f"❌ Error refreshing token: {response}")
                return None
                
        except Exception as e:
            print(f"❌ Exception in token refresh: {str(e)}")
            return None
    
    def validate_token(self, access_token: str) -> bool:
        """
        Validate access token by making a test API call
        """
        try:
            fyers = fyersModel.FyersModel(client_id=self.client_id, token=access_token)
            
            # Test with profile API call
            response = fyers.get_profile()
            
            if response['s'] == 'ok':
                print(f"✅ Token is valid")
                return True
            else:
                print(f"❌ Token validation failed: {response}")
                return False
                
        except Exception as e:
            print(f"❌ Exception in token validation: {str(e)}")
            return False
    
    def logout_token(self, access_token: str) -> bool:
        """
        Logout and invalidate access token
        API Doc: https://myapi.fyers.in/docsv3/#operation/logout
        """
        try:
            fyers = fyersModel.FyersModel(client_id=self.client_id, token=access_token)
            
            response = fyers.logout()
            
            if response['s'] == 'ok':
                print(f"✅ Successfully logged out and invalidated token")
                return True
            else:
                print(f"❌ Error during logout: {response}")
                return False
                
        except Exception as e:
            print(f"❌ Exception during logout: {str(e)}")
            return False

def demo_complete_authentication_flow():
    """Demonstrate complete authentication flow"""
    
    print("🔐 FYERS API v3 - Complete Authentication Demo")
    print("=" * 60)
    
    # Load config
    try:
        with open("../config.json", "r") as f:
            config = json.load(f)
        
        client_id = config["client_id"]
        secret_key = config["secret_key"]
    except:
        print("❌ Please ensure config.json exists with client_id and secret_key")
        return
    
    auth = FyersAuthentication(client_id, secret_key)
    
    # Step 1: Generate authorization URL
    print("\n📋 Step 1: Generate Authorization URL")
    auth_url = auth.step1_generate_auth_url()
    
    if auth_url:
        print("\n💡 Please complete the authorization in your browser")
        print("💡 Copy the auth_code from the redirect URL")
        
        # In real implementation, you would get this from the redirect URL
        # auth_code = input("Enter the auth_code: ")
        # 
        # # Step 2: Generate access token
        # print("\n📋 Step 2: Generate Access Token")
        # tokens = auth.step2_generate_access_token(auth_code)
        # 
        # if tokens:
        #     # Step 3: Validate token
        #     print("\n📋 Step 3: Validate Token")
        #     auth.validate_token(tokens['access_token'])
        #     
        #     # Step 4: Refresh token (if needed)
        #     print("\n📋 Step 4: Refresh Token")
        #     new_tokens = auth.step3_refresh_access_token(tokens['refresh_token'])
        
        print("\n✅ Authentication flow demonstration completed!")

if __name__ == "__main__":
    demo_complete_authentication_flow()
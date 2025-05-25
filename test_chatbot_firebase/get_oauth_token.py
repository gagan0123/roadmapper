import os
from google_auth_oauthlib.flow import Flow
from google.oauth2.credentials import Credentials
import google.auth.transport.requests
import json

# This script helps in the initial OAuth 2.0 flow to get user credentials.
# 1. Run this script.
# 2. It will print an authorization URL. Open this URL in your browser.
# 3. Sign in with your Google account and grant the requested permissions.
# 4. Your browser will be redirected to a URL (likely showing an error because no server is running there).
#    Copy the ENTIRE redirected URL from your browser's address bar.
# 5. Paste this full URL back into the script when prompted.
# 6. The script will attempt to fetch the tokens and save them to 'user_credentials.json'.

CLIENT_SECRETS_FILE = os.path.join(os.path.dirname(__file__), '..', 'private', 'client_secrets.json')
# Ensure your client_secrets.json is in the '../private/' directory relative to this script.
# Or provide an absolute path: os.path.join('/path/to/your/project/root', 'private', 'client_secrets.json')

SCOPES = ['https://www.googleapis.com/auth/drive.readonly', 'https://www.googleapis.com/auth/presentations.readonly']
REDIRECT_URI = 'http://127.0.0.1:7860/callback' # Must be an Authorized redirect URI in your GCP OAuth Client ID settings

# Save the credentials in the same directory as this script
SAVED_CREDENTIALS_FILE = os.path.join(os.path.dirname(__file__), 'user_credentials.json')

def main():
    # IMPORTANT: Allow insecure transport for local development only!
    # This is because our REDIRECT_URI is http://127.0.0.1, not https.
    # Remove this or ensure HTTPS in production.
    os.environ['OAUTHLIB_INSECURE_TRANSPORT'] = '1'

    # Create flow instance to manage the OAuth 2.0 Authorization Grant Flow steps.
    flow = Flow.from_client_secrets_file(
        CLIENT_SECRETS_FILE,
        scopes=SCOPES,
        redirect_uri=REDIRECT_URI
    )

    # Create an DWD URL to send to the user.
    authorization_url, state = flow.authorization_url(
        access_type='offline',  # offline access so that you can refresh an access token without user interaction.
        include_granted_scopes='true',  # This is recommended for offline access.
        prompt='consent'  # Force the consent screen to ensure refresh_token
    )

    print('Please open this URL in your browser to authorize the application:')
    print(authorization_url)
    print('-'*70)

    # Get the authorization server's response from the user.
    redirected_url = input('After authorization, your browser will redirect. It might show an error. \nPlease paste the FULL redirected URL here: ')
    print('-'*70)


    try:
        # Exchange the authorization code for an access token and refresh token.
        flow.fetch_token(authorization_response=redirected_url)
    except Exception as e:
        print(f"Error fetching token: {e}")
        print("Please ensure the redirected URL was pasted correctly and includes the 'code=' parameter.")
        print("Also, ensure the REDIRECT_URI in this script matches an Authorized redirect URI in your GCP OAuth Client ID settings.")
        return

    credentials = flow.credentials

    # Save the credentials for the next run
    creds_data = {
        'token': credentials.token,
        'refresh_token': credentials.refresh_token,
        'token_uri': credentials.token_uri,
        'client_id': credentials.client_id,
        'client_secret': credentials.client_secret,
        'scopes': credentials.scopes
    }
    with open(SAVED_CREDENTIALS_FILE, 'w') as f:
        json.dump(creds_data, f)

    print(f'Credentials saved to {SAVED_CREDENTIALS_FILE}')
    print('You can now use these credentials to initialize the Google Drive API client for the authenticated user.')
    print(f"Access Token: {credentials.token[:30]}...")
    if credentials.refresh_token:
        print(f"Refresh Token: {credentials.refresh_token[:30]}...")
    else:
        print("Refresh Token: Not received (this might happen if it was already granted and stored by Google for this client).")

if __name__ == '__main__':
    main() 
import sys
import os

# --- Apply shared global test configurations ---
try:
    from . import global_test_config # Assuming it's in the same 'tests' package
    global_test_config.apply_global_test_configurations()
except ImportError:
    # Fallback for running the script directly where relative import might fail
    # This assumes global_test_config.py is in the same directory as this script
    import global_test_config
    global_test_config.apply_global_test_configurations()
# --- End shared global test configurations ---

from google.cloud import aiplatform
from google.cloud import storage
from google.oauth2 import service_account

def check_environment():
    """Check if required environment variables are set."""
    required_vars = [
        'GOOGLE_CLOUD_PROJECT',
        'GOOGLE_APPLICATION_CREDENTIALS'
    ]
    
    missing_vars = []
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        print("Missing required environment variables:")
        for var in missing_vars:
            print(f"- {var}")
        return False
    
    return True

def verify_credentials():
    """Verify that the service account credentials are valid."""
    try:
        credentials = service_account.Credentials.from_service_account_file(
            os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
        )
        print("✓ Service account credentials are valid")
        return True
    except Exception as e:
        print(f"✗ Error loading service account credentials: {e}")
        return False

def verify_project_access():
    """Verify access to the Google Cloud project."""
    try:
        project_id = os.getenv('GOOGLE_CLOUD_PROJECT')
        print(f"\nVerifying access to project: {project_id}")
        
        # Initialize Vertex AI
        aiplatform.init(project=project_id)
        print("✓ Vertex AI initialized successfully")
        
        # Test Storage access
        storage_client = storage.Client(project=project_id)
        buckets = list(storage_client.list_buckets())
        print(f"✓ Successfully listed {len(buckets)} storage buckets")
        
        return True
    except Exception as e:
        print(f"✗ Error accessing project: {e}")
        return False

def main():
    print("Testing Google Cloud Setup...")
    
    # Check environment variables
    if not check_environment():
        sys.exit(1)
    
    # Verify credentials
    if not verify_credentials():
        sys.exit(1)
    
    # Verify project access
    if not verify_project_access():
        sys.exit(1)
    
    print("\n✓ All tests passed! Your Google Cloud setup is working correctly.")

if __name__ == "__main__":
    main()
import os
from github import Github
from google.cloud import storage
from dotenv import load_dotenv

def test_github_token():
    """Test GitHub token access."""
    token = os.getenv('GITHUB_TOKEN')
    if not token:
        print("✗ GitHub token not found in environment variables")
        return False
    
    try:
        # Initialize GitHub client
        g = Github(token)
        
        # Test API access
        user = g.get_user()
        print(f"✓ GitHub token is valid. Authenticated as: {user.login}")
        return True
    except Exception as e:
        print(f"✗ Error with GitHub token: {e}")
        return False

def test_gcs_bucket():
    """Test GCS bucket access."""
    bucket_name = os.getenv('GCS_BUCKET_NAME')
    if not bucket_name:
        print("✗ GCS bucket name not found in environment variables")
        return False
    
    try:
        # Initialize Storage client
        storage_client = storage.Client()
        
        # Test bucket access
        bucket = storage_client.bucket(bucket_name)
        if not bucket.exists():
            print(f"✗ Bucket {bucket_name} does not exist")
            return False
        
        print(f"✓ Successfully accessed GCS bucket: {bucket_name}")
        return True
    except Exception as e:
        print(f"✗ Error accessing GCS bucket: {e}")
        return False

def main():
    # Load environment variables
    load_dotenv()
    
    print("Testing GitHub and GCS Setup...\n")
    
    # Test GitHub token
    print("Testing GitHub token...")
    github_ok = test_github_token()
    
    # Test GCS bucket
    print("\nTesting GCS bucket access...")
    gcs_ok = test_gcs_bucket()
    
    # Summary
    print("\nSummary:")
    print(f"GitHub Token: {'✓ OK' if github_ok else '✗ Failed'}")
    print(f"GCS Bucket: {'✓ OK' if gcs_ok else '✗ Failed'}")
    
    if github_ok and gcs_ok:
        print("\n✓ All tests passed! Your GitHub and GCS setup is working correctly.")
    else:
        print("\n✗ Some tests failed. Please check the errors above.")

if __name__ == "__main__":
    main() 
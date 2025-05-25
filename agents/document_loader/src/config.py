import os
from dotenv import load_dotenv
from pydantic import BaseModel
from pydantic_settings import BaseSettings

# Load environment variables
load_dotenv()

# Debug: Print current working directory and .env file location
print(f"Current working directory: {os.getcwd()}")
print(f"Looking for .env file in: {os.path.join(os.getcwd(), '.env')}")
print(f".env file exists: {os.path.exists(os.path.join(os.getcwd(), '.env'))}")

# Debug: Print environment variables
print("\nEnvironment variables:")
for var in ['GOOGLE_CLOUD_PROJECT', 'GOOGLE_APPLICATION_CREDENTIALS', 'GITHUB_TOKEN', 'GCS_BUCKET_NAME']:
    print(f"{var}: {os.getenv(var)}")

class Config(BaseSettings):
    # Google Cloud Configuration
    google_cloud_project: str
    google_application_credentials: str
    
    # GitHub Configuration
    github_token: str
    
    # Vector Search Configuration
    index_name: str = 'document-search-index'
    gcs_bucket_name: str
    
    class Config:
        env_file = '.env'
        env_file_encoding = 'utf-8'
        env_prefix = ''  # No prefix for environment variables
        case_sensitive = False  # Allow case-insensitive matching
        extra = 'ignore'  # Ignore extra fields in the .env file

    @classmethod
    def validate(cls) -> bool:
        """Validate that all required configuration is present."""
        required_vars = [
            'GOOGLE_CLOUD_PROJECT',
            'GOOGLE_APPLICATION_CREDENTIALS',
            'GITHUB_TOKEN',
            'GCS_BUCKET_NAME'
        ]
        
        missing_vars = [var for var in required_vars if not getattr(cls, var)]
        
        if missing_vars:
            print("Missing required environment variables:")
            for var in missing_vars:
                print(f"- {var}")
            return False
            
        return True 
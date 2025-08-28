import os
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables from .env file
load_dotenv()

def get_openai_client():
    """
    Create and return an OpenAI client with configuration from environment variables.
    """
    api_key = os.getenv('OPENAI_API_KEY')
    base_url = os.getenv('OPENAI_BASE_URL')
    
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in environment variables")
    if not base_url:
        raise ValueError("OPENAI_BASE_URL not found in environment variables")
    
    return OpenAI(
        api_key=api_key,
        base_url=base_url
    )

# Create a global client instance
client = get_openai_client()

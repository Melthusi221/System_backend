# app/utils/config.py
import os
from dotenv import load_dotenv

def load_environment():
    """Load environment variables from .env file"""
    load_dotenv()
    return {
        'API_USERNAME': os.getenv('API_USERNAME'),
        'API_PASSWORD': os.getenv('API_PASSWORD')
    }